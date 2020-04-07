# -*- coding: utf-8 -*-
"""
@author: ryuichi takanobu
"""

import torch
import torch.nn as nn

class HybridValue(nn.Module):
    def __init__(self, cfg):
        super(HybridValue, self).__init__()

        self.net_sys_s = nn.Sequential(nn.Linear(cfg.s_dim, cfg.hs_dim),
                                       nn.ReLU(),
                                       nn.Linear(cfg.hs_dim, cfg.hs_dim),
                                       nn.Tanh())
        self.net_usr_s = nn.Sequential(nn.Linear(cfg.s_dim_usr, cfg.hs_dim),
                                       nn.ReLU(),
                                       nn.Linear(cfg.hs_dim, cfg.hs_dim),
                                       nn.Tanh())
        
        self.net_sys = nn.Sequential(nn.Linear(cfg.hs_dim, cfg.h_dim),
                                     nn.ReLU(),
                                     nn.Linear(cfg.h_dim, 1))
        self.net_usr = nn.Sequential(nn.Linear(cfg.hs_dim, cfg.h_dim),
                                     nn.ReLU(),
                                     nn.Linear(cfg.h_dim, 1))
        self.net_global = nn.Sequential(nn.Linear(cfg.hs_dim+cfg.hs_dim, cfg.h_dim),
                                        nn.ReLU(),
                                        nn.Linear(cfg.h_dim, 1))
        
    def forward(self, s, character):
        if character == 'sys':
            h_s_sys = self.net_sys_s(s)
            v = self.net_sys(h_s_sys)
        elif character == 'usr':
            h_s_usr = self.net_usr_s(s)
            v = self.net_usr(h_s_usr)
        elif character == 'global':
            h_s_usr = self.net_usr_s(s[0])
            h_s_sys = self.net_sys_s(s[1])
            h = torch.cat([h_s_usr, h_s_sys], -1)
            v = self.net_global(h)
        else:
            raise NotImplementedError('Unknown character {}'.format(character))
        return v.squeeze(-1)
