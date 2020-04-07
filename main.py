# -*- coding: utf-8 -*-
"""
@author: ryuichi takanobu
"""
import sys
import time
import logging
from utils import get_parser, init_logging_handler
from datamanager import DataManager
from config import MultiWozConfig
from torch import multiprocessing as mp
from policy import Policy
from learner import Learner
from controller import Controller
from agenda import UserAgenda
from rule import SystemRule

def worker_policy_sys(args, manager, config):
    init_logging_handler(args.log_dir, '_policy_sys')
    agent = Policy(None, args, manager, config, 0, 'sys', True)
    
    best = float('inf')
    for e in range(args.epoch):
        agent.imitating(e)
        best = agent.imit_test(e, best)

def worker_policy_usr(args, manager, config):
    init_logging_handler(args.log_dir, '_policy_usr')
    agent = Policy(None, args, manager, config, 0, 'usr', True)
    
    best = float('inf')
    for e in range(args.epoch):
        agent.imitating(e)
        best = agent.imit_test(e, best)

def make_env(data_dir, config):
    controller = Controller(data_dir, config)
    return controller
    
def make_env_rule(data_dir, config):
    env = SystemRule(data_dir, config)
    return env

def make_env_agenda(data_dir, config):
    env = UserAgenda(data_dir, config)
    return env

if __name__ == '__main__':
    parser = get_parser()
    argv = sys.argv[1:]
    args, _ = parser.parse_known_args(argv)
    
    if args.config == 'multiwoz':
        config = MultiWozConfig()
    else:
        raise NotImplementedError('Config of the dataset {} not implemented'.format(args.config))

    init_logging_handler(args.log_dir)
    logging.debug(str(args))
    
    try:
        mp = mp.get_context('spawn')
    except RuntimeError:
        pass
    
    if args.pretrain:
        logging.debug('pretrain')
        
        manager = DataManager(args.data_dir, config)
        processes = []
        process_args = (args, manager, config)
        processes.append(mp.Process(target=worker_policy_sys, args=process_args))
        processes.append(mp.Process(target=worker_policy_usr, args=process_args))
        for p in processes:
            p.start()
        
        for p in processes:
            p.join()
       
    elif args.test:
        logging.debug('test')
        logging.disable(logging.DEBUG)
    
        agent = Learner(make_env, args, config, 1, infer=True)
        agent.load(args.load)
        agent.evaluate(args.test_case)
        
        # test system policy with agenda
        env = make_env_agenda(args.data_dir, config)
        agent.evaluate_with_agenda(env, args.test_case)

        # test user policy with rule
        env = make_env_rule(args.data_dir, config)
        agent.evaluate_with_rule(env, args.test_case)
                
    else: # training
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        logging.debug('train {}'.format(current_time))
    
        agent = Learner(make_env, args, config, args.process)
        best = agent.load(args.load)

        for i in range(args.epoch):
            agent.update(args.batchsz_traj, i)
            # validation
            best = agent.update(args.batchsz, i, best)
            current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            logging.debug('epoch {} {}'.format(i, current_time))
