import torch
import torch.nn as nn
from torchvision import transforms

'''
    input x: m*C*H*W
        m个C*H*W的图片

'''
class L1Agg(nn.Moduule):
    # Convolution
    def __init__(self):
        super(L1Agg, self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(2,1,7,padding=3))
    
    def get_dist(x1, x2):   # C*H*W tensor
        dist = 0.0f
        for i in range(x1.shape[0]):
            for j in range(x1.shape[1]):
                for k in range(x1.shape[2]):
                    d = float(x1[i][j][k]) - float(x2[i][j][k])
                    dist = dist + d * d
        return dist

    def forward(self, x):
        # x is tensor, x 被resnet特征提取过了
        S = [i for i in range(len(x))]  # index of X rest to be aggregated
        X = [x[i] for i in range(len(x))]   # list of C*H*W tensor
        while len(S) > 1:
            minI_1 = 0
            minI_2 = 0
            minL = float("inf")
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    L = get_dist(X[i], X[j])# L = 计算距离
                    if L < minL:
                        minI_1 = i
                        minI_2 = j
                        minL = L
            
            C = []  # c*H*W
            for i in range(len(x[0])):  # 遍历每个channel
                stack = torch.stack(X[minI_1][i], X[minI_2][i]) # shape[2,h,w]
                ts = torch.unsqueeze(stack, dim=0)  # shape[1,2,h,w]
                C.append(self.layer(ts)[0][0])

            S.remove(minI_1)
            S.remove(minI_2)
            C = torch.stack(C)
            S.append(len(X))
            X.append(C)
                
            # 找出两个最近的图片 卷积（手写/Conv2d）结果放进x里
        return S[0]

'''
class L2Agg(nn.Module):
    # Convolution -> Batch Norm -> ReLU -> Convolution
    def __init__(self):
        super(L2Agg, self).__init__()
        self.layer = nn.Sequential(
                        nn.Conv2d(),
                        nn.BatchNorm2d(),
                        nn.ReLU(inplace=True),
                        nn.Conv2d())

    def forward(self, x):
        return x = self.layer(x)

class L3Agg(nn.Module):
    # Convolution -> Batch Norm -> ReLU --> Convolution -> Batch Norm -> ReLU -> Convolution
    def __init__(self):
        super(L3Agg, self).__init__()

        self.layer = nn.Sequential(
                        nn.Conv2d(inchannel, outchannel, 1, stride=stride, bias = False),
                        nn.BatchNorm2d(),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(),
                        nn.BatchNorm2d(),
                        nn.ReLU(inplace=True),
                        nn.Conv2d())
 
    def forward(self, x):
        return self.layer(x)
'''
