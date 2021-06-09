import pytest
import pandas
import torch
import numpy as np


def test_agg_666_calc_dist():
    t = []
    for i in range(1, 5):  # batch
        a = []
        for j in range(1, 6):  # items
            b = []
            for k in range(2*j+i, 2*j + 5+i):  # cwh
                b.append(k)
            a.append(b)
        t.append(a)

    # [batch, items, cwh]
    t = torch.tensor(t)
    # print(t)
    sq_t = torch.sum(t ** 2, dim=2)
    # print(sq_t)
    sum_sq_t = sq_t.unsqueeze(2)
    sum_sq_t2 = sq_t.unsqueeze(1)
    # print(sum_sq_t)
    # print(sum_sq_t2)
    sq_a = sum_sq_t + sum_sq_t2 - 2 * torch.bmm(t, t.transpose(1, 2))
    print(sq_a)


def test_agg_666_calc_mindist():
    t = torch.randn(4, 5, 5)
    print(t)
    t = t.view(4, 5*5)
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
    cwh = 3 * 2 * 4
    t = x.reshape(batch, items, cwh)
    sq_t = torch.sum(t ** 2, dim=2)
    sum_sq_t1 = sq_t.unsqueeze(2)
    sum_sq_t2 = sq_t.unsqueeze(1)
    sq_dist = sum_sq_t1 + sum_sq_t2 - 2 * torch.bmm(t, t.transpose(1, 2))
    sq_dist = sq_dist + \
        torch.diag_embed(torch.ones(items) * np.inf).repeat(batch, 1, 1)
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
    # print(index_i)
    # print(index_i.shape)
    # X_l/X_r [batch, c, w, h]
    X_l = torch.gather(t, dim=1, index=index_i).reshape(batch, c, w, h)
    #reshape(batch, c, w, h)
    #X_r = torch.gather(x, dim=1, index=index_j)
    X_l = X_l.unsqueeze(1)
    print(x)
    print(X_l)

    x = torch.cat((x, X_l), dim=1)
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
    print(mask_row)
    mask_col = mask.unsqueeze(2).repeat(1, 1, items)
    print(mask_col)
    mask_matrix = mask_row + mask_col
    print(mask_matrix)
