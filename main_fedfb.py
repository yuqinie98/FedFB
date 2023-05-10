#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import random

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate, LocalUpdateBuffer
from models.Nets import MLP, CNNMnist, CNNCifar, SmallCNN, LargeCNN
from models.Fed import FedAvg, FedBuff
from models.test import test_img
import time

import argparse

fix_seed = 1234
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='FedFB')
# federated arguments
parser.add_argument('--rounds', type=int, default=10, help="rounds of training")
parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
parser.add_argument('--bs', type=int, default=128, help="test batch size")
parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

# model arguments
parser.add_argument('--model', type=str, default='cnn', help='model name')
parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                    help='comma-separated kernel size to use for convolution')
parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
parser.add_argument('--max_pool', type=str, default='True',
                    help="Whether use max pooling rather than strided convolutions")

# other arguments
parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
parser.add_argument('--verbose', action='store_true', help='verbose print')
parser.add_argument('--seed', type=int, default=1234, help='random seed (default: 1)')
parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
parser.add_argument('--local_buffer', action='store_true', help='local buffer')
parser.add_argument('--global_buffer', action='store_true', help='global buffer')
parser.add_argument('--top_percent', type=float, default=0.2, help='ratio of filtered data')


if __name__ == '__main__':
    t0 = time.time()
    
    # parse args
    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'small_cnn' and args.dataset == 'mnist':
        net_glob = SmallCNN(args=args).to(args.device)
    elif args.model == 'large_cnn' and args.dataset == 'mnist':
        net_glob = LargeCNN(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    # copy weights
    global_gradient = net_glob.state_dict()

    # training
    loss_trains = []
    acc_trains, acc_tests = [], []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    local_cumulative_gradients = [{} for i in range(args.num_users)]
    global_cumulative_gradient = {}

    if args.all_clients: 
        print("Aggregation over all clients")
        local_gradients = [global_gradient for i in range(args.num_users)]
    for iter in range(args.rounds): 
        t1 = time.time()
        net_glob.train()
        
        loss_locals = []
        if not args.all_clients:
            local_gradients = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdateBuffer(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local_gradient, local_cumulative_gradients[idx], loss = local.train(net=copy.deepcopy(net_glob), cumulative_gradient=local_cumulative_gradients[idx])
            if args.all_clients:
                local_gradients[idx] = copy.deepcopy(local_gradient)
            else:
                local_gradients.append(copy.deepcopy(local_gradient))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        # global_gradient = FedAvg(local_gradients)
        global_w = copy.deepcopy(net_glob).state_dict()
        global_w, global_cumulative_gradient = FedBuff(global_w, global_cumulative_gradient, local_gradients, args)

        # copy weight to net_glob
        net_glob.load_state_dict(global_w)
        
        # testing
        net_glob.eval()
        acc_train, loss_train = test_img(net_glob, dataset_train, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        
        #loss_avg = sum(loss_locals) / len(loss_locals)
        loss_trains.append(loss_train)
        acc_trains.append(acc_train)
        acc_tests.append(acc_test)

        # print
        t2 = time.time()
        print('Round {:d}, Average loss {:.3f}, Training accuracy: {:.2f}, Testing accuracy: {:.2f}, Round Time {:.3f}s, Total Time {:.3f}s'.format(iter, loss_train, acc_train, acc_test, t2-t1, t2-t0))

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_trains)), loss_trains)
    plt.ylabel('train_loss')
    plt.savefig('./save_figs/fed_{}_{}_{}_C{}_E{}_B{}_alpha{}_iid{}_localB{}_globalB{}.png'.format(args.dataset, 
            args.model, args.rounds, args.frac, args.local_ep, args.local_bs, args.top_percent, args.iid, args.local_buffer, args.global_buffer))
    
    # save data
    loss_trains = np.array(loss_trains)
    acc_trains = np.array(acc_trains)
    acc_tests = np.array(acc_tests)
    np.savez('./save_results/fed_{}_{}_{}_C{}_E{}_B{}_alpha{}_iid{}_localB{}_globalB{}.npz'.format(args.dataset, args.model, args.rounds, args.frac, args.local_ep, 
            args.local_bs, args.top_percent, args.iid, args.local_buffer, args.global_buffer), loss_trains=loss_trains, acc_trains=acc_trains, acc_tests=acc_test)

