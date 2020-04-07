# -*- coding: utf-8 -*-
"""
@author: ryuichi takanobu
"""
import time
import logging
import os
import numpy as np
import argparse
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str, default='log', help='Logging directory')
    parser.add_argument('--data_dir', type=str, default='multiwoz', help='Data directory')
    parser.add_argument('--save_dir', type=str, default='model_multi', help='Directory to store model')
    parser.add_argument('--load', type=str, default='', help='File name to load trained model')
    parser.add_argument('--pretrain', type=bool, default=False, help='Set to pretrain')
    parser.add_argument('--test', type=bool, default=False, help='Set to inference')
    parser.add_argument('--config', type=str, default='multiwoz', help='Dataset to use')
    parser.add_argument('--test_case', type=int, default=1000, help='Number of test cases')
    parser.add_argument('--save_per_epoch', type=int, default=4, help="Save model every XXX epoches")
    parser.add_argument('--print_per_batch', type=int, default=200, help="Print log every XXX batches")

    parser.add_argument('--epoch', type=int, default=48, help='Max number of epoch')
    parser.add_argument('--process', type=int, default=8, help='Process number')
    parser.add_argument('--batchsz', type=int, default=32, help='Batch size')
    parser.add_argument('--batchsz_traj', type=int, default=512, help='Batch size to collect trajectories')
    parser.add_argument('--policy_weight_sys', type=float, default=2.5, help='Pos weight on system policy pretraining')
    parser.add_argument('--policy_weight_usr', type=float, default=4, help='Pos weight on user policy pretraining')
    parser.add_argument('--lr_policy', type=float, default=1e-3, help='Learning rate of dialog policy')
    parser.add_argument('--lr_vnet', type=float, default=3e-5, help='Learning rate of value network')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 penalty)')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted factor')
    parser.add_argument('--clip', type=float, default=10, help='Gradient clipping')
    parser.add_argument('--interval', type=int, default=400, help='Update interval of target network')

    return parser

def discard(dic, key, value=None):
    if key in dic:
        if value is None or dic[key] == value:
            del(dic[key])

def init_session(key, cfg):
    # shared info
    turn_data = {}
    turn_data['others'] = {'session_id':key, 'turn':0, 'terminal':False, 'change':False}
    turn_data['sys_action'] = dict()
    turn_data['user_action'] = dict()
    
    # belief & goal state 
    turn_data['belief_state'] = {}
    turn_data['goal_state'] = {}
    for domain in cfg.belief_domains:
        turn_data['belief_state'][domain] = {}
        turn_data['goal_state'][domain] = {}
        
    # user goal
    session_data = {}
    for domain in cfg.belief_domains:
        session_data[domain] = {}
    
    return turn_data, session_data

def init_goal(goal, state, off_goal, cfg):
    for domain in cfg.belief_domains:
        if domain in off_goal and off_goal[domain]:
            domain_data = off_goal[domain]
            # constraint
            if 'info' in domain_data:
                for slot, value in domain_data['info'].items():
                    slot = cfg.map_inverse[domain][slot]
                    # single slot value for user goal
                    inform_da = domain+'-'+slot
                    if inform_da in cfg.inform_da_usr:
                        goal[domain][slot] = value
                        state[domain][slot] = value
            if 'fail_info' in domain_data and domain_data['fail_info']:
                goal[domain]['final'] = {}
                for slot, value in domain_data['fail_info'].items():
                    slot = cfg.map_inverse[domain][slot]
                    # single slot value for user goal
                    inform_da = domain+'-'+slot
                    if inform_da in cfg.inform_da_usr:
                        goal[domain]['final'][slot] = goal[domain][slot]
                        goal[domain][slot] = value
                        state[domain][slot] = value
                        
            # booking
            if 'book' in domain_data:
                goal[domain]['book'] = True
                for slot, value in domain_data['book'].items():
                    if slot in cfg.map_inverse[domain]:
                        slot = cfg.map_inverse[domain][slot]
                        # single slot value for user goal
                        inform_da = domain+'-'+slot
                        if inform_da in cfg.inform_da_usr:
                            goal[domain][slot] = value
                            state[domain][slot] = value
            if 'fail_book' in domain_data and domain_data['fail_book']:
                if 'final' not in goal[domain]:
                    goal[domain]['final'] = {}
                for slot, value in domain_data['fail_book'].items():
                    if slot in cfg.map_inverse[domain]:
                        slot = cfg.map_inverse[domain][slot]
                        # single slot value for user goal
                        inform_da = domain+'-'+slot
                        if inform_da in cfg.inform_da_usr:
                            goal[domain]['final'][slot] = goal[domain][slot]
                            goal[domain][slot] = value
                            state[domain][slot] = value
                        
            # request
            if 'reqt' in domain_data:
                for slot in domain_data['reqt']:
                    slot = cfg.map_inverse[domain][slot]
                    request_da = domain+'-'+slot
                    if request_da in cfg.request_da_usr:
                        goal[domain][slot] = '?'
                        state[domain][slot] = '?'
                        
def reload(state, goal, domain):
    state[domain] = {}
    for key in goal[domain]:
        if key != 'final':
            state[domain][key] = goal[domain][key]
    if 'final' in goal[domain]:
        for key in goal[domain]['final']:
            goal[domain][key] = goal[domain]['final'][key]
            state[domain][key] = goal[domain][key]
        del(goal[domain]['final'])

def init_logging_handler(log_dir, extra=''):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    stderr_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('{}/log_{}.txt'.format(log_dir, current_time+extra))
    logging.basicConfig(handlers=[stderr_handler, file_handler])
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

def to_device(data):
    if type(data) == dict:
        for k, v in data.items():
            data[k] = v.to(device=DEVICE)
    else:
        for idx, item in enumerate(data):
            data[idx] = item.to(device=DEVICE)
    return data

def check_constraint(slot, val_usr, val_sys):
    try:
        if slot == 'arrive':
            val1 = int(val_usr.split(':')[0]) * 100 + int(val_usr.split(':')[1])
            val2 = int(val_sys.split(':')[0]) * 100 + int(val_sys.split(':')[1])
            if val1 < val2:
                return True
        elif slot == 'leave':
            val1 = int(val_usr.split(':')[0]) * 100 + int(val_usr.split(':')[1])
            val2 = int(val_sys.split(':')[0]) * 100 + int(val_sys.split(':')[1])
            if val1 > val2:
                return True
        else:
            if val_usr != val_sys:
                return True
        return False
    except:
        return False

def state_vectorize(state, config, db, noisy=False):
    """
    state: dict_keys(['user_action', 'sys_action', 'select_entity', 'belief_state', 'others']) 
    state_vec: [user_act, last_sys_act, inform, request, book, degree, entropy]
    """
    user_act = np.zeros(len(config.da_usr))
    for da in state['user_action']:
        user_act[config.da2idx_u[da]] = 1.
    
    last_sys_act = np.zeros(len(config.da))
    for da in state['sys_action']:
        last_sys_act[config.da2idx[da]] = 1.
    
    inform = np.zeros(len(config.inform_da))
    request = np.zeros(len(config.request_da))
    for domain in state['belief_state']:
        for slot, value in state['belief_state'][domain].items():
            key = domain+'-'+slot
            if value == '?':
                if key in config.request2idx:
                    request[config.request2idx[key]] = 1.
            else:
                if key in config.inform2idx:
                    inform[config.inform2idx[key]] = 1.
    
    # select entity
    book = np.zeros(len(config.belief_domains))
    for domain in state['belief_state']:
        if 'booked' in state['belief_state'][domain]:
            book[config.domain2idx[domain]] = 1.
    
    degree, entropy = db.pointer(state['belief_state'], config.mapping, config.db_domains, config.requestable, noisy)
            
    final = 1. if state['others']['terminal'] else 0.
    
    state_vec = np.r_[user_act, last_sys_act, inform, request, book, degree, final]
    assert len(state_vec) == config.s_dim
    return state_vec

def action_vectorize(action, config):
    act_vec = np.zeros(config.a_dim)
    for da in action:
        act_vec[config.da2idx[da]] = 1
    return act_vec

def state_vectorize_user(state, config, current_domain):
    """
    state: dict_keys(['user_action', 'sys_action', 'user_goal', 'goal_state', 'others']) 
    state_vec: [sys_act, last_user_act, inform, request, focus, inconsistency, nooffer]
    """    
    sys_act = np.zeros(len(config.da))
    for da in state['sys_action']:
        sys_act[config.da2idx[da]] = 1.

    last_user_act = np.zeros(len(config.da_usr))
    for da in state['user_action']:
        last_user_act[config.da2idx_u[da]] = 1.
        
    inform = np.zeros(len(config.inform_da_usr))
    request = np.zeros(len(config.request_da_usr))
    for domain in state['goal_state']:
        if domain in state['invisible_domains']:
            continue
        for slot, value in state['goal_state'][domain].items():
            key = domain+'-'+slot
            if value == '?':
                if key in config.request2idx_u:
                    request[config.request2idx_u[key]] = 1.
            else:
                if key in config.inform2idx_u and slot in state['user_goal'][domain]\
                    and state['user_goal'][domain][slot] != '?':
                    inform[config.inform2idx_u[key]] = 1.
    
    focus = np.zeros(len(config.belief_domains))
    if current_domain:
        focus[config.domain2idx[current_domain]] = 1.
    
    inconsistency = np.zeros(len(config.inform_da_usr))
    nooffer = np.zeros(len(config.belief_domains))
    for da, value in state['sys_action'].items():
        domain, intent, slot, p = da.split('-')
        if intent in ['inform', 'recommend', 'offerbook', 'offerbooked']:
            key = domain+'-'+slot
            if key in config.inform2idx_u and slot in state['user_goal'][domain]:
                refer = state['user_goal'][domain][slot]
                if refer != '?' and check_constraint(slot, refer, value):
                    inconsistency[config.inform2idx_u[key]] = 1.
        if intent in ['nooffer', 'nobook'] and current_domain:
            nooffer[config.domain2idx[current_domain]] = 1.
            
    state_vec = np.r_[sys_act, last_user_act, inform, request, inconsistency, nooffer]
    assert len(state_vec) == config.s_dim_usr
    return state_vec

def action_vectorize_user(action, terminal, config):
    act_vec = np.zeros(config.a_dim_usr)
    for da in action:
        act_vec[config.da2idx_u[da]] = 1
    if terminal:
        act_vec[-1] = 1
    return act_vec
    
def reparameterize(mu, logvar):
    std = (0.5*logvar).exp()
    eps = torch.randn_like(std)
    return eps.mul(std) + mu
