#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
from utils.filter import filter_top_gradients


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedBuff(global_w, global_cumulative_gradient, w, args):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    
    for name, param in w_avg.items():
        if args.global_buffer:
            if name not in global_cumulative_gradient:
                global_cumulative_gradient[name] = torch.zeros_like(param)
            global_cumulative_gradient[name] += param
            filtered_gradient = filter_top_gradients(global_cumulative_gradient[name], args.top_percent)
            updated_mask = filtered_gradient != 0
            param.copy_(filtered_gradient + global_w[name])
            global_cumulative_gradient[name] = global_cumulative_gradient[name] * (~updated_mask)
        else:
            filtered_gradient = filter_top_gradients(param, args.top_percent)
            param.copy_(filtered_gradient + global_w[name])

    return w_avg, global_cumulative_gradient