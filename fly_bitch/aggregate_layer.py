import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torchvision import transforms
from loguru import logger
import torch.autograd.profiler as profiler

import numpy as np


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

class Agg_666(nn.Module):
    def __init__(self, level="L1", logging=False):
        super(Agg_666, self).__init__()
        if level == "L1":
            self.layer = Agg2(L1Agg(), logging=logging)
        else:
            raise "Not supported"
        self.logging = logging

    def forward(self, x):
        # (batch, items, c, w, h)
        batch, items, c, w, h = x.shape[0]
        cwh = c * w * h
        mask = torch.zeros(items).repeat(batch, 1)
        src_ = torch.ones(batch).unsqueeze(1) * np.inf   # static const
        # items 个东东要聚合 items-1 次
        n = items
        for _ in range(1, n):
            # (batch, items, c * w * h)
            t = x.reshape(batch, items, cwh)
            # 使用下面方法
            # https://blog.csdn.net/Answer3664/article/details/115006057
            # 这里把a和b都记为t
            sq_t = torch.sum(t ** 2, dim=2)
            sum_sq_t1 = sq_t.unsqueeze(2)
            sum_sq_t2 = sq_t.unsqueeze(1)
            sq_dist = sum_sq_t1 + sum_sq_t2 - 2 * torch.bmm(t, t.transpose(1,2))
            # sq_dist [batch, items, items] , sq_dist[9][2][3] 表示第10组中第3和第4个item之间的距离
            # 由于sq_dist[k][i][i] = 0，但要忽略它，需要把对角线设置为inf，即加上对角线为inf的矩阵
            sq_dist = sq_dist + torch.diag_embed(torch.ones(items) * np.inf).repeat(batch, 1, 1)
            mask_row = mask.unsqueeze(1).repeat(1, items, 1)
            mask_col = mask.unsqueeze(2).repeat(1, 1, items)
            mask_matrix = mask_row + mask_col
            sq_dist = sq_dist + mask_matrix

            sq_dist = sq_dist.view(batch, items * items)
            # indices shape[batch]
            _, indices = sq_dist.min(1)
            # 0 <= k < batch
            # i_k = indices[k] // items
            # j_k = indices[k] % items
            # 即对于第k+1组对i和j聚合
            # [batch, 1]
            index_i = (indices // items).unsqueeze(1)
            index_j = (indices % items).unsqueeze(1)
            # 为了调用gather函数，需要把index_i/j复制扩充为[batch, 1, cwh]
            index_ii = index_i.unsqueeze(1).repeat(1, 1, cwh)
            index_jj = index_j.unsqueeze(1).repeat(1, 1, cwh)
            # X_l/X_r [batch, c, w, h]
            X_l = torch.gather(t, dim=1, index=index_ii).reshape(batch, c, w, h)
            X_r = torch.gather(t, dim=1, index=index_jj).reshape(batch, c, w, h)
            # X即为新聚合成的item
            # [batch, c, w, h]
            X = self.layer(X_l, X_r)
            # [batch, 1, c, w, h]
            X = X.unsqueeze(1)
            # [batch, items + 1, c, w, h]
            x = torch.cat((x, X), dim=1)

            # 对mask增加，然后往x里头加入X
            # 对mask添加inf表示删去被用过的index_i和index_j
            add_ = torch.zeros(1).repeat(batch, 1)
            mask = torch.cat((mask, add_), dim=1)
            mask = mask.scatter(1, index_i, src_)
            mask = mask.scatter(1, index_j, src_)
            # mask [batch, items + 1]   每组一个mask，inf代表已经被聚合过了
            # [[0, 0, .., inf, .. , inf, 0, inf, ..]
            #  [inf, 0, inf, .., inf, 0, inf, .., 0]
            #  ...
            #  [0, inf, 0, .., 0, inf, .., 0, inf]]

            items = x.shape[1]
        # end aggregate

        return x[:, items-1]
        


def test_agg_666_calc_dist():
    t = []
    for i in range(1, 5): # batch
        a = []
        for j in range(1, 6): # items
            b = []
            for k in range(2*j+i, 2*j + 5+i): # cwh
                b.append(k)
            a.append(b)
        t.append(a)

    # [batch, items, cwh]
    t = torch.tensor(t)
    #print(t)
    sq_t = torch.sum(t ** 2, dim=2)
    #print(sq_t)
    sum_sq_t = sq_t.unsqueeze(2)
    sum_sq_t2 = sq_t.unsqueeze(1)
    #print(sum_sq_t)
    #print(sum_sq_t2)
    sq_a = sum_sq_t + sum_sq_t2 - 2 * torch.bmm(t, t.transpose(1,2))
    print(sq_a)
    

def test_agg_666_calc_mindist():
    t = torch.randn(4, 5, 5)
    print(t)
    t = t.view(4,5*5)
    print(t)
    b = t.min(1)
    print(b[0])
    print(b[1])
    # get index = col * i + j
    # i = index // col
    # j = index % col

def test_agg_666_diag_inf():
    # [batch, items, items]
    t = torch.randn(4, 5, 5)
    # 把对角线设置为最大值
    a = torch.diag_embed(torch.ones(5) * np.inf).repeat(4, 1, 1)
    b = t + a
    c = b.view(4, 5*5).min(1)
    print(b)
    print(c)

def test_agg_666_agg():
    # [batch, items, c, w, h]
    x = torch.randn(4, 5, 3, 2, 4)
    batch, items, c, w, h = x.shape
    cwh = 3 * 2* 4
    t = x.reshape(batch, items, cwh)
    sq_t = torch.sum(t ** 2, dim=2)
    sum_sq_t1 = sq_t.unsqueeze(2)
    sum_sq_t2 = sq_t.unsqueeze(1)
    sq_dist = sum_sq_t1 + sum_sq_t2 - 2 * torch.bmm(t, t.transpose(1,2))
    sq_dist = sq_dist + torch.diag_embed(torch.ones(items) * np.inf).repeat(batch, 1, 1)
    print(sq_dist.shape)
    sq_dist = sq_dist.view(batch, items * items)
    # indices shape[batch]
    _, indices = sq_dist.min(1)
            # 0 <= k < batch
            # i_k = indices[k] // items
            # j_k = indices[k] % items
            # 即对于第k+1组对i和j聚合
    index_i = indices // items
    index_j = indices % items

    print(indices)
    print(index_i)
    print(index_j)
    index_i = index_i.unsqueeze(1).unsqueeze(1).repeat(1, 1, cwh)
    #print(index_i)
    #print(index_i.shape)
    # X_l/X_r [batch, c, w, h]
    X_l = torch.gather(t, dim=1, index=index_i).reshape(batch, c, w, h)
    #reshape(batch, c, w, h)
    #X_r = torch.gather(x, dim=1, index=index_j)
    X_l = X_l.unsqueeze(1)
    print(x)
    print(X_l)
    
    x = torch.cat((x, X_l), dim = 1)
    print(x)


def test_agg_666_mask():
    # [batch, items, c, w, h]
    x = torch.randn(4, 5, 3, 2, 4)
    batch, items, c, w, h = x.shape
    # [batch, items, items]
    # a = torch.diag_embed(torch.ones(items) * np.inf).repeat(batch, 1, 1)
    
    # 扩张为 [batch, items + 1, items + 1]
    a = torch.diag_embed(torch.ones(items + 1) * np.inf).repeat(batch, 1, 1)
    # init [batch, items]
    mask = torch.zeros(items).repeat(batch, 1)

    # add to [batch, items + 1]
    src_ = torch.ones(batch).unsqueeze(1) * np.inf   # static
    print(src_)
    index_i = torch.tensor([4, 1, 2, 3]).unsqueeze(1)
    add_ = torch.zeros(1).repeat(batch, 1)
    mask = torch.cat((mask, add_), dim=1)
    mask = mask.scatter(1, index_i, src_)
    
    print(mask.shape)
    print(mask)
    # 按行repeat
    items = mask.shape[1]
    # mask [batch, items]
    mask_row = mask.unsqueeze(1).repeat(1, items, 1)
    print (mask_row)
    mask_col = mask.unsqueeze(2).repeat(1, 1, items)
    print (mask_col)
    mask_matrix = mask_row + mask_col
    print(mask_matrix)
    




test_agg_666_mask()
#test_agg_666_agg()
#test_agg_666_diag_inf()
#test_agg_666_calc_mindist()
#test_agg_666_calc_dist()