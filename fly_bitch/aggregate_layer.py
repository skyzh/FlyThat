import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torchvision import transforms
from loguru import logger
import torch.autograd.profiler as profiler


class L1Agg(nn.Module):
    '''
    input x: m*C*H*W
        m个C*H*W的图片
    '''

    def __init__(self):
        super(L1Agg, self).__init__()
        self.layer = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        return self.layer(x)


class Agg2(nn.Module):
    # layer 是一个LxAgg
    def __init__(self, layer, logging=False):
        super(Agg2, self).__init__()
        self.layer = layer
        self.logging = logging

    def forward(self, x1, x2):  # c * h * w
        C = []
        if self.logging:
            """
            Merge: torch.Size([16, 512, w, h]) torch.Size([16, 512, w, h])
            """
            logger.info(f'Merge: {x1.shape} {x2.shape}')
        stack = torch.stack((x1, x2), dim=2)

        if self.logging:
            """
            After stack: torch.Size([16, 512, 2, w, h])
            """
            logger.info(f'After stack: {stack.shape}')

        batch, channels, items, w, h = stack.shape
        stack = stack.reshape(batch * channels, items, w, h)
        stack = self.layer(stack)
        if self.logging:
            """
            After stack: torch.Size([16, 512, 2, w, h])
            """
            logger.info(f'After conv: {stack.shape}')

        stack = stack.squeeze()
        stack = stack.reshape(batch, channels, w, h)

        if self.logging:
            logger.info(f'Finally: {stack.shape}')
        return stack


class Agg(nn.Module):
    def __init__(self, level="L1", logging=False):
        super(Agg, self).__init__()
        if level == "L1":
            self.layer = Agg2(L1Agg(), logging=logging)
        else:
            raise "Not supported"
        self.logging = logging

    def forward(self, x):
        # (batch, items, c, w, h) -> (items, batch, c, w, h)
        x = torch.swapaxes(x, 0, 1)
        if self.logging:
            logger.info(f'After swap axes: {x.shape}')
        items = x.shape[0]
        X = x[0]
        for i in range(1, items):
            X = self.layer(X, x[i])
        return X
