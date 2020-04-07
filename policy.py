# -*- coding: utf-8 -*-
"""
@author: ryuichi takanobu
"""
import os
import logging

import torch
import torch.nn as nn
from torch import optim

from utils import to_device 
from evaluator import MultiWozEvaluator
from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiDiscretePolicy(nn.Module):
    def __init__(self, cfg, character='sys'):
        super(MultiDiscretePolicy, self).__init__()
        
        if character == 'sys':
            self.net = nn.Sequential(nn.Linear(cfg.s_dim, cfg.h_dim),
                                     nn.ReLU(),
                                     nn.Linear(cfg.h_dim, cfg.h_dim),
                                     nn.ReLU(),
                                     nn.Linear(cfg.h_dim, cfg.a_dim))
        elif character == 'usr':
            self.net = nn.Sequential(nn.Linear(cfg.s_dim_usr, cfg.h_dim),
                                     nn.ReLU(),
                                     nn.Linear(cfg.h_dim, cfg.h_dim),
                                     nn.ReLU(),
                                     nn.Linear(cfg.h_dim, cfg.a_dim_usr))
        else:
            raise NotImplementedError('Unknown character {}'.format(character))

    def forward(self, s):
        # [b, s_dim] => [b, a_dim]
        a_weights = self.net(s)

        return a_weights
    
    def select_action(self, s, sample=True):
        """
        :param s: [s_dim]
        :return: [a_dim]
        """
        # forward to get action probs
        # [s_dim] => [a_dim]
        a_weights = self.forward(s)
        a_probs = torch.sigmoid(a_weights)
        
        # [a_dim] => [a_dim, 2]
        a_probs = a_probs.unsqueeze(1)
        a_probs = torch.cat([1-a_probs, a_probs], 1)
        
        # [a_dim, 2] => [a_dim]
        a = a_probs.multinomial(1).squeeze(1) if sample else a_probs.argmax(1)
        
        return a
    
    def batch_select_action(self, s, sample=False):
        """
        :param s: [b, s_dim]
        :return: [b, a_dim]
        """
        # forward to get action probs
        # [b, s_dim] => [b, a_dim]
        a_weights = self.forward(s)
        a_probs = torch.sigmoid(a_weights)
        
        # [b, a_dim] => [b, a_dim, 2]
        a_probs = a_probs.unsqueeze(2)
        a_probs = torch.cat([1-a_probs, a_probs], 2)
        
        # [b, a_dim, 2] => [b*a_dim, 2] => [b*a_dim, 1] => [b*a_dim] => [b, a_dim]
        a = a_probs.reshape(-1, 2).multinomial(1).squeeze(1).reshape(a_weights.shape) if sample else a_probs.argmax(2)
        
        return a
    
    def get_log_prob(self, s, a):
        """
        :param s: [b, s_dim]
        :param a: [b, a_dim]
        :return: [b, 1]
        """
        # forward to get action probs
        # [b, s_dim] => [b, a_dim]
        a_weights = self.forward(s)
        a_probs = torch.sigmoid(a_weights)
        
        # [b, a_dim] => [b, a_dim, 2]
        a_probs = a_probs.unsqueeze(-1)
        a_probs = torch.cat([1-a_probs, a_probs], -1)
        
        # [b, a_dim, 2] => [b, a_dim]
        trg_a_probs = a_probs.gather(-1, a.unsqueeze(-1)).squeeze(-1)
        log_prob = torch.log(trg_a_probs)
        
        return log_prob.sum(-1, keepdim=True)
        

class Policy(object):
    def __init__(self, env_cls, args, manager, cfg, process_num, character, pre=False, infer=False):
        """
        :param env_cls: env class or function, not instance, as we need to create several instance in class.
        :param args:
        :param manager:
        :param cfg:
        :param process_num: process number
        :param character: user or system
        :param pre: set to pretrain mode
        :param infer: set to test mode
        """

        self.process_num = process_num
        self.character = character

        # initialize envs for each process
        self.env_list = []
        for _ in range(process_num):
            self.env_list.append(env_cls())

        # construct policy and value network
        self.policy = MultiDiscretePolicy(cfg, character).to(device=DEVICE)
        
        if pre:
            self.print_per_batch = args.print_per_batch
            from dbquery import DBQuery
            db = DBQuery(args.data_dir, cfg)
            self.data_train = manager.create_dataset_policy('train', args.batchsz, cfg, db, character)
            self.data_valid = manager.create_dataset_policy('valid', args.batchsz, cfg, db, character)
            self.data_test = manager.create_dataset_policy('test', args.batchsz, cfg, db, character)
            if character == 'sys':
                pos_weight = args.policy_weight_sys * torch.ones([cfg.a_dim]).to(device=DEVICE)
            elif character == 'usr':
                pos_weight = args.policy_weight_usr * torch.ones([cfg.a_dim_usr]).to(device=DEVICE)
            else:
                raise Exception('Unknown character')
            self.multi_entropy_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.evaluator = MultiWozEvaluator()

        self.save_dir = args.save_dir + '/' + character if pre else args.save_dir
        self.save_per_epoch = args.save_per_epoch
        self.optim_batchsz = args.batchsz
        self.policy.eval()
        
        self.gamma = args.gamma
        self.policy_optim = optim.RMSprop(self.policy.parameters(), lr=args.lr_policy, weight_decay=args.weight_decay)
        self.writer = SummaryWriter()

    def policy_loop(self, data):
        s, target_a = to_device(data)
        a_weights = self.policy(s)
        
        loss_a = self.multi_entropy_loss(a_weights, target_a)
        return loss_a

    def imitating(self, epoch):
        """
        pretrain the policy by simple imitation learning (behavioral cloning)
        """
        self.policy.train()
        a_loss = 0.
        for i, data in enumerate(self.data_train):
            self.policy_optim.zero_grad()
            loss_a = self.policy_loop(data)
            a_loss += loss_a.item()
            loss_a.backward()
            self.policy_optim.step()
            
            if (i+1) % self.print_per_batch == 0:
                a_loss /= self.print_per_batch
                logging.debug('<<dialog policy {}>> epoch {}, iter {}, loss_a:{}'.format(self.character, epoch, i, a_loss))
                a_loss = 0.
        
        if (epoch+1) % self.save_per_epoch == 0:
            self.save(self.save_dir, epoch)
        self.policy.eval()
    
    def imit_test(self, epoch, best):
        """
        provide an unbiased evaluation of the policy fit on the training dataset
        """        
        a_loss = 0.
        for i, data in enumerate(self.data_valid):
            loss_a = self.policy_loop(data)
            a_loss += loss_a.item()
            
        a_loss /= len(self.data_valid)
        logging.debug('<<dialog policy {}>> validation, epoch {}, loss_a:{}'.format(self.character, epoch, a_loss))
        if a_loss < best:
            logging.info('<<dialog policy {}>> best model saved'.format(self.character))
            best = a_loss
            self.save(self.save_dir, 'best')
            
        a_loss = 0.
        for i, data in enumerate(self.data_test):
            loss_a = self.policy_loop(data)
            a_loss += loss_a.item()
            
        a_loss /= len(self.data_test)
        logging.debug('<<dialog policy {}>> test, epoch {}, loss_a:{}'.format(self.character, epoch, a_loss))
        self.writer.add_scalar('pretrain/dialogue_policy_{}/test'.format(self.character), a_loss, epoch)
        return best
    
    def save(self, directory, epoch):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save(self.policy.state_dict(), directory + '/' + str(epoch) + '_pol.mdl')

        logging.info('<<dialog policy {}>> epoch {}: saved network to mdl'.format(self.character, epoch))

    def load(self, filename):
        
        policy_mdl = filename + '_pol.mdl'
        if os.path.exists(policy_mdl):
            self.policy.load_state_dict(torch.load(policy_mdl))
            logging.info('<<dialog policy {}>> loaded checkpoint from file: {}'.format(self.character, policy_mdl))
