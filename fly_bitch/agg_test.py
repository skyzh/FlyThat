import os.path
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
from torch.autograd import gradcheck

from agg import *
from fea import *


torch.set_printoptions(profile="full")

transform = transforms.Compose([
        transforms.ToTensor(),  # h*w*c转变为c*h*w，正则化为0-1之间
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),    # 正则化为-1到1之间
    ])

root = os.path.abspath('.')
root = os.path.join(root, "data/train")
img_paths = ["59462101751_s.bmp","59469557468_s.bmp","59461911165_s.bmp","59465191869_s.bmp","59467726628_s.bmp","59466495936_s.bmp"]
imgs = []
for img_path in img_paths:
    img = Image.open(os.path.join(root, img_path))
    img = transform(img)
    imgs.append(img)

img_tensor = torch.stack(imgs)  # [m,c,h,w]
img_tensor.requires_grad = True
print(img_tensor.shape)
agg = Agg(Agg2(L1Agg()))
fe = fea()
#conv = nn.Conv2d(3, 64, 7, stride = 7, padding = 3, bias = False)
#y = conv(img_tensor)
#print(y.shape)
#y = fe(img_tensor)
#print(y.shape)
#y.backward(torch.ones_like(y))
#print(img_tensor.grad.shape)


##train(img_tensor)
y = agg(img_tensor)
print(y.shape)
y.backward(torch.ones_like(y))
print(img_tensor.grad.shape)

'''
print(img_tensor.shape)
p1 = img_tensor[0]
p2 = img_tensor[1]
'''
'''
stack = torch.stack((p1, p2)) # shape[2,h,w]
ts = torch.unsqueeze(stack, dim=0)  # shape[1,2,h,w]
ts.requires_grad = True

l1agg = L1Agg()
agg2 = Agg2(L1Agg())
p1.requires_grad = True
p2.requires_grad = True

y_l1agg = agg2(p1, p2)
print(y_l1agg.shape)
y_l1agg.backward(torch.ones_like(y_l1agg))
print(p1.grad)
'''
#ts.requires_grad = True
#y = l1agg(ts)
#y.backward(torch.ones_like(y))
#print(ts.grad.shape)
#'''