import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torchvision import transforms
from . import fea

'''
    input x: m*C*H*W
        m个C*H*W的图片
'''


class L1Agg(nn.Module):
    # Convolution
    def __init__(self):
        super(L1Agg, self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3))

    def forward(self, x):
        return self.layer(x)


class Agg2(nn.Module):
    # layer 是一个LxAgg
    def __init__(self, layer):
        super(Agg2, self).__init__()
        self.layer = layer

    def forward(self, x1, x2):  # c * h * w
        C = []
        for i in range(len(x1)):  # 遍历每个channel
            stack = torch.stack((x1[i], x2[i]))  # shape[2,h,w]
            ts = torch.unsqueeze(stack, dim=0)  # shape[1,2,h,w]
            r = self.layer(ts).squeeze()    # shape[h, w]
            C.append(r)

        return torch.stack(C)


class Agg(nn.Module):
    def __init__(self, level="L1"):
        super(Agg, self).__init__()
        if level == "L1":
            self.layer = Agg2(L1Agg())
        else:
            self.layer = Agg2(L1Agg())

    def forward(self, x):
        X = x[0]

        for i in range(1, len(x)):
            X = self.layer(X, x[i])

        return X
