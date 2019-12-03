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
        print(result)
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

# df = pd.DataFrame.from_csv("data/train.csv")
def apply_flops(df_):
    flops_ex = []
    params_ex = []
    archs = (df_['arch_and_hp'])
    for i in range(archs.shape[0]):
        nn_str = parse_layer(archs[i])
        flops, params = flops_param_calculator(nn_str)
        flops_ex.append(flops)
        params_ex.append(params)
    return pd.DataFrame(flops_ex)

# print(len(flops_ex), len(params_ex))
# print(flops_ex)
# print(params_ex)
