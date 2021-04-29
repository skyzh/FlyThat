from torch import nn
import torch
from .model import NeuralNetwork
from . import utils
from .dataset import DrosophilaTrainImageDataset
from loguru import logger
import argparse
from .feature_extractor import feature_transformer
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

import os
import time
from tensorboardX import SummaryWriter
import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score, roc_auc_score
from .dataset import MAX_LABELS
import torch.autograd.profiler as profiler


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        
        loss.backward()
        optimizer.step()
            
        loss, current = loss.item(), batch * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        



def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    model.eval()

    score_list = []
    label_list = []
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            # TODO: implement this
            # [batch, 30]
            outputs = model(X)

            score_list.extend(outputs.detach().cpu().numpy())
            label_list.extend(y.cpu().numpy())
            

    # [batch * 30]
    score_array = np.concatenate(score_list)
    score_array_ = np.where(score_array > 0.5, 1, 0)
    label_onehot = np.concatenate(label_list)

    print(score_array)
    print(label_onehot)    

    
    # auc average=macro
    auc_ = -1.0
    try:
        auc_ = roc_auc_score(label_onehot, score_array)
    except ValueError:
        # 出现label_onehot全是0的情况
        pass
    
    # f1
    f1_macro = f1_score(label_onehot, score_array_, average='macro')
    f1_micro = f1_score(label_onehot, score_array_, average='micro')

    logger.info(
        f"Test Error: \n AUC {auc_}, f1_macro {f1_macro}, f1_micro {f1_micro} \n"
    )

    return auc_, f1_macro, f1_micro


def main(argv):
    parser = argparse.ArgumentParser(description='Generate PyTorch Dataset')
    parser.add_argument('tensorboard_logs_path', type=str, nargs='?', help='Path of tensorboard logs',
                        default=str(Path() / 'fly_bitch/runs/logs'))
    parser.add_argument('model_path', type=str, nargs='?', help='Path of model',
                        default=str(Path() / 'fly_bitch/model/model.pkl'))
    parser.add_argument('data_path', type=str, nargs='?', help='Path of unzip data',
                        default=str(Path() / 'data'))
    parser.add_argument('batch_size', type=int, nargs='?', help='Batch size',
                        default=8)
    args=parser.parse_args(argv)
    tensorboard_logs_path=args.tensorboard_logs_path
    # '' 经过Path会被解析成 './', exists就会判定True
    model_path=args.model_path
    data_path=Path(args.data_path)
    logger.info(f"using tensorboard logs path '{tensorboard_logs_path}'")
    logger.info(f"using model path '{model_path}'")
    logger.info(f"using data path '{data_path}'")

    device=utils.get_device()
    logger.info(f'Using {device} device')
    # 这里就限制GPU保存，GPU上加载
    model=NeuralNetwork()
    if os.path.exists(model_path):
        logger.info(f"loading model from '{model_path}'")
        model=torch.load(model_path)
    
    model.to(device)
    # Enable logging if you want to view internals of model shape
    # model = NeuralNetwork(logging=True)
    loss_fn=nn.BCELoss()
    optimizer=torch.optim.SGD(model.parameters(), lr=1e-3)
    epochs=20

    dataset=DrosophilaTrainImageDataset(
        Path(data_path), feature_transformer())
    dataset_length=len(dataset)
    train_length=int(dataset_length * 0.8)
    test_length=dataset_length - train_length
    train_dataset, test_dataset=torch.utils.data.random_split(
        dataset, (train_length, test_length))
    train_dataloader=DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader=DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True)

    writer=SummaryWriter(tensorboard_logs_path)

    # seconds.xxxxx
    last_time=time.time()
    for t in range(epochs):
        logger.info(f"Epoch {t+1}, time {last_time} s\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        auc_, f1_macro, f1_micro=test(test_dataloader, model, loss_fn, device)
        writer.add_scalar('auc', auc_, global_step=t)
        writer.add_scalar('f1_macro', f1_macro, global_step=t)
        writer.add_scalar('f1_micro', f1_micro, global_step=t)

        now_time=time.time()
        if now_time - last_time >= 1800.0:
            last_time=now_time
            # 很离谱的是加了文件夹就会提示找不到路径
            torch.save(
                model, 'model_{}_{}_{}.pkl'.format(str(auc_), str(f1_macro), str(f1_micro)))

    logger.info("Done!")
