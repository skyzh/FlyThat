import os.path

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision
from torch.autograd import Variable

import numpy as np
from PIL import Image

import os

torch.set_printoptions(profile="full")

#backbone = backbone_utils.resnet_fpn_backbone('resnet50', True)
#features = list(backbone.children())[:-1] # 去掉最后的fpn层, 得到resnet的2,3,4层输出
#model = nn.Sequential(*features)   
#model = model.to('cpu')

def feature_extractor(img_path):
    # 宽320 * 高128
    # 位深度 24
    # H*W*C = 128*320*3
    transform = transforms.Compose([
        transforms.ToTensor(),  # h*w*c转变为c*h*w，正则化为0-1之间
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),    # 正则化为-1到1之间
    ])
    img = Image.open(img_path)
    img = transform(img)

    print(img)



    #x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    #print(x)
    #y = model(x)
    #return y

# ======================test=========================
root = os.path.abspath('.')
dire = "data/train/0026345835_s.bmp"
feature_extractor(os.path.join(root, dire))
# ===================================================
#for key, value in y.items():
#    print(key, value)