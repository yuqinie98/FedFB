#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import copy
from utils.filter import filter_top_gradients


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    
class LocalUpdateBuffer(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net, cumulative_gradient):
        original_net = copy.deepcopy(net).state_dict()
        
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        '''
        with torch.no_grad():
            for name, param in net.named_parameters():
                if param.requires_grad:
                    original_param = original_net[name]
                    local_gradient = param.data - original_param.data
                    if self.buffer:
                        if name not in cumulative_gradient:
                            cumulative_gradient[name] = torch.zeros_like(original_param.data)
                        cumulative_gradient[name] += local_gradient
                        filtered_gradient = filter_top_gradients(cumulative_gradient[name], self.top_percent)
                        updated_mask = filtered_gradients != 0
                        #param.copy_(filtered_gradient + original_param.data)
                        param.copy_(filtered_gradient)
                        cumulative_gradient[name] = cumulative_gradient[name] * (~updated_mask)
                    else:
                        filtered_gradient = filter_top_gradients(local_gradient, top_percent)
                        #param.copy_(filtered_gradient + original_param.data)
                        param.copy_(filtered_gradient)
                        '''
        
        for name, param in net.state_dict().items():
            original_param = original_net[name]
            local_gradient = param - original_param
            if self.args.local_buffer:
                if name not in cumulative_gradient:
                    cumulative_gradient[name] = torch.zeros_like(original_param)
                cumulative_gradient[name] += local_gradient
                filtered_gradient = filter_top_gradients(cumulative_gradient[name], self.args.top_percent)
                updated_mask = filtered_gradient != 0
                param.copy_(filtered_gradient)
                cumulative_gradient[name] = cumulative_gradient[name] * (~updated_mask)
            else:
                filtered_gradient = filter_top_gradients(local_gradient, self.args.top_percent)
                param.copy_(filtered_gradient)
                        
        return net.state_dict(), cumulative_gradient, sum(epoch_loss) / len(epoch_loss)
    
