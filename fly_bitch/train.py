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


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            # TODO: implement this
    test_loss /= size
    correct /= size
    logger.info(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct, test_loss


def main(args):
    parser = argparse.ArgumentParser(description='Generate PyTorch Dataset')
    parser.add_argument('tensorboard_logs_path', type=str, nargs='?', help='Path of tensorboard logs',
                        default=str(Path() / 'fly_bitch/runs/logs'))
    parser.add_argument('model_path', type=str, nargs='?', help='Path of model',
                        default=str(Path() / 'fly_bitch/model/model.pkl'))
    parser.add_argument('data_path', type=str, nargs='?', help='Path of unzip data',
                        default=str(Path() / 'data'))
    args = parser.parse_args(args)
    tensorboard_logs_path = args.tensorboard_logs_path
    # '' 经过Path会被解析成 './', exists就会判定True
    model_path = args.model_path
    data_path = Path(args.data_path)
    logger.info(f"using tensorboard logs path '{tensorboard_logs_path}'")
    logger.info(f"using model path '{model_path}'")
    logger.info(f"using data path '{data_path}'")

    device = utils.get_device()
    logger.info(f'Using {device} device')
    # 这里就限制GPU保存，GPU上加载
    model = NeuralNetwork()
    if os.path.exists(model_path):
        model = torch.load(model_path)
        model.to(device)
    # Enable logging if you want to view internals of model shape
    # model = NeuralNetwork(logging=True)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    epochs = 5

    dataset = DrosophilaTrainImageDataset(
        Path(data_path), feature_transformer())
    dataset_length = len(dataset)
    train_length = int(dataset_length * 0.8)
    test_length = dataset_length - train_length
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, test_length))
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    writer = SummaryWriter(tensorboard_logs_path)

    # seconds.xxxxx
    last_time = time.time()
    for t in range(epochs):
        logger.info(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        correct, test_loss = test(test_dataloader, model, loss_fn, device)
        writer.add_scalar('correct', correct, global_step=t)
        writer.add_scalar('test_loss', test_loss, global_step=t)

        now_time = time.time()
        if now_time - last_time >= 1800.0:
            last_time = now_time
            torch.save(model, 'fly_bitch/model/model_{}_{}.pkl'.format(str(correct), str(test_loss)))

    logger.info("Done!")
