import re
from functools import reduce
from torch import nn as nn
from thop import profile
import torch
from torch.nn import Sequential, ReLU, LeakyReLU, SELU, Linear, Conv2d, BatchNorm1d, BatchNorm2d, Dropout, Dropout2d, Tanh, Softmax, MaxPool2d, Flatten
import pandas as pd
import numpy as np

def parse_struct(archline):
    ReLU = archline.count(": ReLU(")
    LeakyReLU = archline.count(": LeakyReLU(")
    SELU = archline.count(": SELU(")
    Linear = archline.count(": Linear(")
    Conv2d = archline.count(": Conv2d(")
    BatchNorm1d = archline.count(": BatchNorm1d(")
    BatchNorm2d = archline.count(": BatchNorm2d(")
    Flatten = archline.count(": Flatten(")
    Dropout = archline.count(": Dropout(")
    Dropout2d = archline.count(": Dropout2d(")
    Tanh = archline.count(": Tanh(")
    Softmax = archline.count(": Softmax(")
    MaxPool2d = archline.count(": MaxPool2d(")
    return ReLU, LeakyReLU, SELU, Linear, Conv2d, BatchNorm1d, BatchNorm2d, Flatten, Dropout, Dropout2d, Tanh, Softmax, MaxPool2d


def parse_order(archline):
    x = [m.start() for m in re.finditer('Conv2d', archline)]
    Conv2d = list(zip(x, np.full(len(x), 0)))
    x = [m.start() for m in re.finditer('Linear', archline)]
    Linear = list(zip(x, np.full(len(x), 1)))
    x = [m.start() for m in re.finditer('BatchNorm1d', archline)]
    BatchNorm1d = list(zip(x, np.full(len(x), 2)))
    x = [m.start() for m in re.finditer('BatchNorm2d', archline)]
    BatchNorm2d = list(zip(x, np.full(len(x), 3)))
    result = Conv2d+Linear+BatchNorm1d+BatchNorm2d
    result = sorted(result, key=lambda x: x[0])
    ret = list(zip(*result))
    if len(ret) != 0:
        ret = ret[1]
    return ret


def calculate_flop(archline):
    result = 0
    Conv2dstart = archline.find(': Conv2d(')
    while(Conv2dstart != -1):
        strstart = Conv2dstart + len(": Conv2d(")
        Conv2dend = archline.find('))', Conv2dstart) + 1
        toparse = archline[strstart:Conv2dend]
        # print(toparse)
        temp1 = re.findall(r'\d+', toparse)
        res = list(map(int, temp1))
        result += reduce(lambda x, y: x*y, res)
        # print(result)
        temp = archline.find(': Conv2d(', Conv2dstart + 1)
        Conv2dstart = temp


def parse_layer(archline):
    Conv2dstart = archline.find('): ')
    flag = 0
    while(Conv2dstart != -1):
        strstart = Conv2dstart
        while(archline[strstart] != '('):
            strstart -= 1
        strstart -= 1
        target = archline[strstart:Conv2dstart+2]
        if (flag == 0):
            archline = archline.replace(target, '')
            flag = 1
        else:
            archline = archline.replace(target, ',')
        Conv2dstart = archline.find('): ')
    return archline


def flops_param_calculator(arch_hp):
    try:
        model = eval(arch_hp)
    except Exception:
        print('Fail to evalute string as nn: ', Exception)
        return
    return profile(model, inputs=(torch.randn(1, 3, 32, 32), ), verbose=False)


# eval to nn
# arch = "Sequential( Conv2d(3, 28, ernel_size=(1, 1), stride=(1, 1)) ,Flatten(), Linear(in_features=28672, out_features=29, bias=True) , BatchNorm1d(29, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) , Tanh() , Tanh() , Tanh() , Dropout(p=0.714631919301416), Linear(in_features=29, out_features=47, bias=True),Dropout(p=0.97788680890307) , Linear(in_features=47, out_features=10, bias=True) ,Tanh() ,Softmax() )"
# model = eval(arch)
# input = torch.randn(1, 3, 32, 32) # CIFAR10 dim
# flops, params = profile(model, inputs=(input, ))
# print(flops, params)

def apply_flops(df_):
    flops_ex = []
    params_ex = []
    archs = (df_['arch_and_hp'])
    for i in range(archs.shape[0]):
        nn_str = parse_layer(archs[i])
        flops, params = flops_param_calculator(nn_str)
        flops_ex.append(flops)
        params_ex.append(params)
    return pd.DataFrame({'tot_flops': params_ex})


def apply_ops_hist(df_):
    ret = np.empty((df_.shape[0], 13))
    archs = (df_['arch_and_hp'])
    for i in range(archs.shape[0]):
        ret[i] = parse_struct(archs[i])
    df_ret = pd.DataFrame(ret)
    df_ret.columns = ['ReLU', 'LeakyReLU', 'SELU', 'Linear', 'Conv2d', 'BatchNorm1d', 'BatchNorm2d', 'Flatten', 'Dropout', 'Dropout2d', 'Tanh', 'Softmax', 'MaxPool2d']
    return df_ret
# print(len(flops_ex), len(params_ex))
# print(flops_ex)
# print(params_ex)


def apply_init_params(df_):
    im = np.array(df_['init_params_mu'], dtype=str)
    ex1 = np.zeros((len(im), 2))
    for i in range(len(im)):
        ex1[i] = np.sum(np.array(eval(im[i])).reshape([-1, 2]), axis=0)

    im = np.array(df_['init_params_std'], dtype=str)
    ex2 = np.zeros((len(im), 2))
    for i in range(len(im)):
        ex2[i] = np.sum(np.array(eval(im[i])).reshape([-1, 2]), axis=0)

    im = np.array(df_['init_params_l2'], dtype=str)
    ex3 = np.zeros((len(im), 2))
    for i in range(len(im)):
        ex3[i] = np.sum(np.array(eval(im[i])).reshape([-1, 2]), axis=0)
    return pd.DataFrame({'init_A_mu': ex1[:,0], 'init_b_mu': ex1[:,1], 'init_A_std': ex2[:,0], 'init_b_std':ex2[:,1], 'init_A_l2':ex3[:,0], 'init_b_l2':ex3[:,1]})


def diff_avg(arr, header):
    dif = np.diff(arr)
    ret = np.empty((dif.shape[0], 7))
    for j in range(dif.shape[0]):
        ret[j] = np.mean(dif[j].reshape((7, 7)), axis=1)
    df_ret = pd.DataFrame(ret)
    df_ret.columns = [header+str(i) for i in range(7)]
    return df_ret
# df = pd.read_csv("data/train-1185.csv")
# feature_all = list(df.columns)
# for i in range(len(feature_all)):
#     print(i, feature_all[i])
# print(diff_avg(df[feature_all[17:67]]))
# diff_avg(df[feature_all[67:117]])
# diff_avg(df[feature_all[117:167]])
# diff_avg(df[feature_all[167:217]])

