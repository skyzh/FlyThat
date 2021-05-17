import torch
from torch.utils.data import Dataset
from .dataset import MAX_LABELS


def generate_weights(dataset):
    count = {}
    N = 0
    for (_, label) in dataset:
        idx = 0
        seq = 1
        for mask in label.int().numpy().tolist():
            idx |= mask * seq
            seq <<= 1

        count[idx] = count.get(idx, 0) + 1

        N = N + 1

    weight_per_labelc = {}
    for key, value in count.items():
        weight_per_labelc[key] = len(count) / float(value)

    weight_per_bag = [0] * N
    for i, (_, label) in enumerate(dataset):
        idx = 0
        seq = 1
        for mask in label.int().numpy().tolist():
            idx |= mask * seq
            seq <<= 1

        weight_per_bag[i] = weight_per_labelc[idx]

    return count, torch.DoubleTensor(weight_per_bag)


def make_weight_sampler(dataset):
    _, weights = generate_weights(dataset)
    return torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

class RemedialDataset(Dataset):
    def __init__(self):
        self.imgs = []
        self.labels = []
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return (self.imgs[idx], self.labels[idx])
    
    def extend_item(self, img, label_onehot):
        self.imgs.append(img)
        self.labels.append(label_onehot)


# IRLbl(y) = max_label_num / label_num[y]
# mean_IR
def make_label_set(dataset):
    # labels_num[i] means the number of the label i which is active in dataset 
    labels_num = [0] * MAX_LABELS
    ninstances = 0  # the number of instances
    for (_, label) in dataset:
        for i, mask in enumerate(label.int().numpy().tolist()):
            labels_num[i] = labels_num[i] + mask

        ninstances = ninstances + 1

    max_label_num = 0.0
    dsum_label_num = 0.0 # sum_i(1 / labels_num[i])
    for num in labels_num:
        dsum_label_num = dsum_label_num + 1 / float(num)
        max_label_num = max(max_label_num, num)
    
    # Known:
    # dsum_label_num = sum(1 / labels_num[y])
    # mean_IR = sum(max_label_num / labels_num[y]) / MAX_LABELS
    #         = max_label_num * sum(1 / labels_num[y]) / MAX_LABELS
    # =>
    mean_IR = max_label_num * dsum_label_num / MAX_LABELS
    IRLbl = [0] * MAX_LABELS
    for i, _ in enumerate(labels_num):
        IRLbl[i] = max_label_num / float(labels_num[i])
    return IRLbl, mean_IR, ninstances


def make_scumble_set(dataset, IRLbl, mean_IR, ninstances):
    SCUMBLEins = [0] * ninstances
    dL = 1 / float(ninstances)
    sum_SCUMBLE = 0
    for i, (_, label) in enumerate(dataset):
        sum_IRLbl_il = 0.0
        mul_IRLbl_il = 1.0
        for l, mask in enumerate(label.int().numpy().tolist()):
            if mask == 1:
                IRLbl_il = IRLbl[l]
                sum_IRLbl_il = sum_IRLbl_il + IRLbl_il
                mul_IRLbl_il = mul_IRLbl_il * IRLbl_il
            else:
                # 论文说此时IRLbl_il=0，这样mul_IRLbl_il不就直接为0了？
                # 很明显没有一个instance是都active，这样SCUMBLE都是1了
                # 所以忽略掉
                tmp = 0

        avg_IRLbl_il = sum_IRLbl_il / float(MAX_LABELS)
        SCUMBLEins[i] = 1 - (mul_IRLbl_il ** dL) / avg_IRLbl_il
        sum_SCUMBLE = sum_SCUMBLE + SCUMBLEins[i]
    
    avg_SCUMBLE = sum_SCUMBLE / ninstances
    return SCUMBLEins, avg_SCUMBLE


# dataset : (imgs, label)
def remedial(dataset):
    # Calculate imbalance levels
    # IRLbl(y) = max_label_num / label_num[y]
    IRLbl, mean_IR, ninstances = make_label_set(dataset)
    # Calculate SCUMBLE
    SCUMBLEins, avg_SCUMBLE = make_scumble_set(dataset, IRLbl, mean_IR, ninstances)
    
    remedial_dataset = RemedialDataset()
    for i, (imgs, label) in enumerate(dataset):  
        if SCUMBLEins[i] > avg_SCUMBLE:
            imgs_1 = imgs.clone()
            label_1 = label.clone()
            for l, _ in enumerate(label.int().numpy().tolist()):
                if IRLbl[l] <= mean_IR:
                    label[l] = 0
                else:
                    label_1[l] = 0
            
            remedial_dataset.extend_item(imgs_1, label_1)

        remedial_dataset.extend_item(imgs, label)

    return remedial_dataset  

            



def main(_argv):
    from pathlib import Path
    from .dataset import DrosophilaTrainImageDataset
    from .feature_extractor import feature_transformer

    dataset = DrosophilaTrainImageDataset(
        Path() / 'data', feature_transformer(), False)
    dataset_length = len(dataset)
    train_length = int(dataset_length * 0.8)
    test_length = dataset_length - train_length
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, test_length))
    count, _ = generate_weights(train_dataset)
    from matplotlib import pyplot as plt
    plt.bar(list(count.keys()), count.values(), color='g')
    plt.show()

    sampler = make_weight_sampler(train_dataset)

    for idx, data in enumerate(sampler):
        if idx > 100:
            break
        print(data)

    print(test_dataset)
    print(len(test_dataset))
    print(test_dataset.__getitem__(2))
    a = test_dataset.__getitem__(2)
    aa = a[0].clone()
    bb = a[1].clone()
    # remedial作为训练集和测试集dataset
    # dataloader 使用 shuffle = True即可？
    remedial_dataset = remedial(test_dataset)
    print(remedial_dataset)
    print(len(remedial_dataset))
    c = remedial_dataset.__getitem__(2)
    print(c[0] == aa)
    print(c[1] == bb)
    
