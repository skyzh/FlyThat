import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torchvision import transforms
from fea import *

'''
    input x: m*C*H*W
        m个C*H*W的图片
'''
class L1Agg(nn.Module):
    # Convolution
    def __init__(self):
        super(L1Agg, self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(2,1,7,padding=3))

    def forward(self, x):
        return self.layer(x)


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

class Agg2(nn.Module):
    # layer 是一个LxAgg
    def __init__(self, layer):
        super(Agg2, self).__init__()
        self.layer = layer

    def forward(self, x1, x2):  # c * h * w
        #return torch.stack((get_666(x1[i], x[2]) for i in range(len(x1))))
        C = []
        for i in range(len(x1)):  # 遍历每个channel
            stack = torch.stack((x1[i], x2[i])) # shape[2,h,w]
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
        self.feature = fea()

    def forward(self, x):
        x = self.feature(x)
        print(x.shape)
        # x is tensor, x 被resnet特征提取过了
        X = x[0]

        for i in range(1, len(x)):
            X = self.layer(X, x[i])
        
        return X


'''
    input x: m*C*H*W
        m个C*H*W的图片

class Agg(Function):
    # layer 是一个Agg2
    def __init(layer):
        self.layer = layer

    def get_dist(x1, x2):   # C*H*W tensor
        dist = 0.0
        for i in range(x1.shape[0]):
            for j in range(x1.shape[1]):
                for k in range(x1.shape[2]):
                    d = float(x1[i][j][k]) - float(x2[i][j][k])
                    dist = dist + d * d
        return dist

    def forward(ctx, x):
        # x is tensor, x 被resnet特征提取过了
        S = [i for i in range(len(x))]  # index of X rest to be aggregated
        X = [x[i] for i in range(len(x))]   # list of C*H*W tensor
        history = []
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

            # c*H*W
            C = self.layer(X[minI_1], X[minI_2])

            S.remove(minI_1)
            S.remove(minI_2)
            C = torch.stack(C)
            S.append(len(X))
            history.append((minI_1, minI_2, len(X)))
            X.append(C)
                
        # X 记录原先存在的和所有聚合成的instance
        # history 记录每个元组(I1,I2,I3), 即X[I1]和X[I2]聚合成X[I3]
        # S 只剩下一个instance, 即为最终聚合成的instance
        ctx.save_for_backward(X, history)
        return X[S[0]]
        return X, history, X[S[0]]

    # 对S[0]的grad
    def backward(ctx, grad):
        # forward 相当于
        X, history = ctx.saved_variables
        gradX = [torch.ones_like(X[0])] * len(X)
        #for


'''

'''
def get_dist(x1, x2):   # C*H*W tensor
        dist = 0.0
        return dist
        for i in range(x1.shape[0]):
            for j in range(x1.shape[1]):
                for k in range(x1.shape[2]):
                    d = float(x1[i][j][k]) - float(x2[i][j][k])
                    dist = dist + d * d
        return dist



def train(x):
    # x经过resnet18
    layer = Agg2(L1Agg())

    S = [i for i in range(len(x))]  # index of X rest to be aggregated
    X = [x[i] for i in range(len(x))]   # list of C*H*W tensor
    history = []
    while len(S) > 1:
        minI_1 = 0
        minI_2 = 0
        minL = float("inf")
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                L = get_dist(X[S[i]], X[S[j]])# L = 计算距离
                if L < minL:
                    minI_1 = S[i]
                    minI_2 = S[j]
                    minL = L
        print("123")
        # c*H*W
        #X[minI_1].requires_grad = True
        #X[minI_2].requires_grad = True
        C = layer(X[minI_1], X[minI_2])
        C.requires_grad = True
        print(C.shape)
        S.remove(minI_1)
        S.remove(minI_2) 
        S.append(len(X))
        history.append((minI_1, minI_2, len(X)))
        X.append(C)
                
    # X 记录原先存在的和所有聚合成的instance
    # history 记录每个元组(I1,I2,I3), 即X[I1]和X[I2]聚合成X[I3]
    # S 只剩下一个instance, 即为最终聚合成的instance
    #ctx.save_for_backward(X, history)
    

    
        Classfication_model(X[S[0]])
        算loss
        ...
        得出Classfication_model的grad
    
    
    grad = torch.ones_like(X[S[0]])

    X[S[0]].backward(grad)
    print(X[history[-1][0]].grad.shape)

'''