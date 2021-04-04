import os
import os.path
import pandas as pd

# python文件同一目录下data文件夹里有这个
root = os.path.abspath('.')
train_dir = os.path.join(root, "data/train.csv")

# 生成map
# dict_train_dir map格式为 
# key: 基因标签， 唯一的
# value: labels : 文件名列表
# eg. 'CG4246662248_s4': ('29', ['91927689633_s.bmp'])
def get_train_img_dir_dict():
    df = pd.read_csv(train_dir)
    dict_train_img_dir = dict(zip(df.values[:,0], zip(df.values[:,1], map(lambda x : x.strip('()').split(',') , df.values[:, 2]))))
    #print(dict_train_img_dir)
    return dict_train_img_dir

# 生成list
# list_train_dir list格式为
# 元组 (labels, 文件名列表)
# eg. ('29', ['91927689633_s.bmp'])
def get_train_img_dir_list():
    df = pd.read_csv(train_dir)
    list_train_img_dir = list(zip(df.values[:,1], map(lambda x : x.strip('()').split(',') , df.values[:, 2])))
    #print(list_train_img_dir)
    return list_train_img_dir

dict_train_img_dir = get_train_img_dir_dict()
list_train_img_dir = get_train_img_dir_list()

def get_image(img_paths):
    img_root = os.path.join(root, 'data/train')
    for img_path in img_paths:
        img_path = os.path.join(img_root, img_path)
        #print(img_path)
        #调用
        # feature_extractor(img_path)

get_image(list_train_img_dir[2][1])

