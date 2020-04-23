# -*- coding: utf-8 -*-
"""
@author: ryuichi takanobu
"""
import random
import torch
from copy import deepcopy
from dbquery import DBQuery
from utils import discard, reload
from evaluator import MultiWozEvaluator

class StateTracker(object):
    def __init__(self, data_dir, config):
        self.time_step = 0
        self.cfg = config
        self.db = DBQuery(data_dir, config)
        self.topic = ''
        self.evaluator = MultiWozEvaluator(data_dir)
        self.lock_evalutor = False
    
    def set_rollout(self, rollout):
        if rollout:
            self.save_time_step = self.time_step
            self.save_topic = self.topic
            self.lock_evalutor = True
        else:
            self.time_step = self.save_time_step
            self.save_topic = self.topic
            self.lock_evalutor = False
    
    def get_entities(self, s, domain):
        origin = s['belief_state'][domain].items()
        constraint = []
        for k, v in origin:
            if v != '?' and k in self.cfg.mapping[domain]:
                constraint.append((self.cfg.mapping[domain][k], v))
        entities = self.db.query(domain, constraint)
        random.shuffle(entities)
        return entities
        
    def update_belief_sys(self, old_s, a):
        """
        update belief/goal state with sys action
        """
        s = deepcopy(old_s)
        a_index = torch.nonzero(a) # get multiple da indices
        
        self.time_step += 1
        s['others']['turn'] = self.time_step
            
        # update sys/user dialog act
        s['sys_action'] = dict()
        
        # update belief part
        das = [self.cfg.idx2da[idx.item()] for idx in a_index]
        das = [da.split('-') for da in das]
        sorted(das, key=lambda x:x[0]) # sort by domain
        
        entities = [] if self.topic == '' else self.get_entities(s, self.topic)
        return_flag = False
        for domain, intent, slot, p in das:
            if domain in self.cfg.belief_domains and domain != self.topic:
                self.topic = domain
                entities = self.get_entities(s, domain)
                
            da = '-'.join((domain, intent, slot, p))
            if intent == 'request':
                s['sys_action'][da] = '?'
            elif intent in ['nooffer', 'nobook'] and self.topic != '':
                return_flag = True
                if slot in s['belief_state'][self.topic] and s['belief_state'][self.topic][slot] != '?':
                    s['sys_action'][da] = s['belief_state'][self.topic][slot]
                else:
                    s['sys_action'][da] = 'none'
            elif slot == 'choice':
                s['sys_action'][da] = str(len(entities))
            elif slot == 'none':
                s['sys_action'][da] = 'none'
            else:
                num = int(p) - 1
                if self.topic and len(entities) > num and slot in self.cfg.mapping[self.topic]:
                    typ = self.cfg.mapping[self.topic][slot]
                    if typ in entities[num]:
                        s['sys_action'][da] = entities[num][typ]
                    else:
                        s['sys_action'][da] = 'none'
                else:
                    s['sys_action'][da] = 'none'
                
                if not self.topic:
                    continue
                if intent in ['inform', 'recommend', 'offerbook', 'offerbooked', 'book']:
                    discard(s['belief_state'][self.topic], slot, '?')
                    if slot in s['user_goal'][self.topic] and s['user_goal'][self.topic][slot] == '?':
                        s['goal_state'][self.topic][slot] = s['sys_action'][da]
            
                # booked
                if intent == 'inform' and slot == 'car': # taxi
                    if 'booked' not in s['belief_state']['taxi']:
                        s['belief_state']['taxi']['booked'] = 'taxi-booked'
                elif intent in ['offerbooked', 'book'] and slot == 'ref': # train
                    if self.topic in ['taxi', 'hospital', 'police']:
                        s['belief_state'][self.topic]['booked'] = f'{self.topic}-booked'
                        s['sys_action'][da] = f'{self.topic}-booked'
                    elif entities:
                        book_domain = entities[0]['ref'].split('-')[0]
                        if 'booked' not in s['belief_state'][book_domain] and entities:
                            s['belief_state'][book_domain]['booked'] = entities[0]['ref']
                            s['sys_action'][da] = entities[0]['ref']
        
        if return_flag:
            for da in s['user_action']:
                d_usr, i_usr, s_usr = da.split('-')
                if i_usr == 'inform' and d_usr == self.topic:
                    discard(s['belief_state'][d_usr], s_usr)
            reload(s['goal_state'], s['user_goal'], self.topic)
        
        if not self.lock_evalutor:
            self.evaluator.add_sys_da(s['sys_action'])
        
        return s
    
    def update_belief_usr(self, old_s, a):
        """
        update belief/goal state with user action
        """
        s = deepcopy(old_s)
        a_index = torch.nonzero(a) # get multiple da indices
        
        self.time_step += 1
        s['others']['turn'] = self.time_step
        s['others']['terminal'] = 1 if (self.cfg.a_dim_usr-1) in a_index else 0
        
        # update sys/user dialog act
        s['user_action'] = dict()
        
        # update belief part
        das = [self.cfg.idx2da_u[idx.item()] for idx in a_index if idx.item() != self.cfg.a_dim_usr-1]
        das = [da.split('-') for da in das]
        if s['invisible_domains']:
            for da in das:
                if da[0] == s['next_available_domain']:
                    s['next_available_domain'] = s['invisible_domains'][0]
                    s['invisible_domains'].remove(s['next_available_domain'])
                    break
        sorted(das, key=lambda x:x[0]) # sort by domain
        
        for domain, intent, slot in das:
            if domain in self.cfg.belief_domains and domain != self.topic:
                self.topic = domain
            
            da = '-'.join((domain, intent, slot))
            if intent == 'request':
                s['user_action'][da] = '?'
                s['belief_state'][self.topic][slot] = '?'
            elif slot == 'none':
                s['user_action'][da] = 'none'
            else:
                if self.topic and slot in s['user_goal'][self.topic] and s['user_goal'][domain][slot] != '?':
                    s['user_action'][da] = s['user_goal'][domain][slot]
                else:
                    s['user_action'][da] = 'dont care'
                    
                if not self.topic:
                    continue
                if intent == 'inform':
                    s['belief_state'][domain][slot] = s['user_action'][da]
                    if slot in s['user_goal'][self.topic] and s['user_goal'][self.topic][slot] != '?':
                        discard(s['goal_state'][self.topic], slot)
        
        if not self.lock_evalutor:
            self.evaluator.add_usr_da(s['user_action'])
        
        return s
    
    def reset(self, random_seed=None):
        """
        Args:
            random_seed (int):
        Returns:
            init_state (dict):
        """
        pass
    
    def step(self, s, sys_a):
        """
        Args:
            s (dict):
            sys_a (vector):
        Returns:
            next_s (dict):
            terminal (bool):
        """
        pass
