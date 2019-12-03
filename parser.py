import re
from functools import reduce
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