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
from .focal_loss import FocalLoss


def result_threshold(score_array, use_maximum_if_none=False, threshold=0.5):
    if use_maximum_if_none:
        data = []
        for row in score_array:
            if np.all(row <= threshold):
                data.append(np.where(row >= np.max(row), 1, 0))
            else:
                data.append(row)
        score_array = np.array(data)
    return np.where(score_array > threshold, 1, 0)


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
        logger.info(f"Train Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device, identifier):
    model.eval()

    score_list = []
    label_list = []
    loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            # TODO: implement this
            # [batch, 30]
            outputs = model(X)
            loss += loss_fn(outputs, y)
            score_list.append(outputs.detach().cpu().numpy())
            label_list.append(y.cpu().numpy())

    score_array = np.concatenate(score_list)
    label_onehot = np.concatenate(label_list)
    # calculate AUC based on continuous value
    auc_ = 0.0
    try:
        auc_ = roc_auc_score(label_onehot, score_array)
    except ValueError:
        pass

    # calculate f1 on discrete value
    score_array = result_threshold(score_array, True)

    # f1
    f1_macro = f1_score(label_onehot, score_array, average='macro')
    f1_micro = f1_score(label_onehot, score_array, average='samples')

    logger.info(
        f"{identifier} Loss {loss}, AUC {auc_}, F1 Macro {f1_macro}, F1 Samples {f1_micro}"
    )

    return auc_, f1_macro, f1_micro, loss


def main(argv):
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--log', type=str, nargs='?', help='Path of tensorboard logs',
                        default=str(Path() / 'runs/logs'))
    parser.add_argument('--model', type=str, nargs='?', help='Model checkpoint path',
                        default=str(Path() / 'runs/model'))
    parser.add_argument('--data', type=str, nargs='?', help='Path of unzip data',
                        default=str(Path() / 'data'))
    parser.add_argument('--batch', type=int, nargs='?', help='Batch size',
                        default=64)
    parser.add_argument('--epoch', type=int, nargs='?', help='Epoch',
                        default=50)
    parser.add_argument('--partial', action='store_true',
                        help='Only use part of the data', default=False)
    args = parser.parse_args(argv)
    tensorboard_logs_path = Path(args.log)
    model_path = Path(args.model)
    data_path = Path(args.data)
    logger.info(
        f"Tensorboard logs will be saved to: '{tensorboard_logs_path}'")
    logger.info(f"Model will be saved to: '{model_path}'")
    logger.info(f"Data will be loaded from: '{data_path}'")

    if model_path.exists():
        logger.warning(
            "There are models inside model path, it's better to remove the directory before training")

    if tensorboard_logs_path.exists():
        logger.warning(
            "There are logs inside log path, it's better to remove the directory before training")

    model_path.mkdir(parents=True, exist_ok=True)
    tensorboard_logs_path.mkdir(parents=True, exist_ok=True)

    if args.partial:
        logger.warning(f"Partial mode enabled, only use first 10 bags")

    device = utils.get_device()
    logger.info(f'Using {device} device')
    # 这里就限制GPU保存，GPU上加载
    # model = NeuralNetwork(logging=True)
    model = NeuralNetwork()
    model = model.to(device)
    loss_fn = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = args.epoch

    dataset = DrosophilaTrainImageDataset(
        Path(data_path), feature_transformer(), args.partial)
    dataset_length = len(dataset)
    train_length = int(dataset_length * 0.8)
    test_length = dataset_length - train_length
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, test_length))
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch, shuffle=True)

    writer = SummaryWriter(tensorboard_logs_path)

    for t in range(epochs):
        logger.info(f"Epoch {t+1}")
        train(train_dataloader, model, loss_fn,
              optimizer, device)

        auc_, f1_macro, f1_micro, loss = test(
            train_dataloader, model, loss_fn, device, 'Train')
        writer.add_scalar('Loss/train', loss, global_step=t)
        writer.add_scalar('AUC/train', auc_, global_step=t)
        writer.add_scalar('F1Macro/train', f1_macro, global_step=t)
        writer.add_scalar('F1Micro/train', f1_micro, global_step=t)

        auc_, f1_macro, f1_micro, loss = test(
            test_dataloader, model, loss_fn, device, 'Test')
        writer.add_scalar('Loss/test', loss, global_step=t)
        writer.add_scalar('AUC/test', auc_, global_step=t)
        writer.add_scalar('F1Macro/test', f1_macro, global_step=t)
        writer.add_scalar('F1Micro/test', f1_micro, global_step=t)
        torch.save(
            model.state_dict(), os.path.join(args.model, 'model_{:02d}_{:.4f}_{:.4f}_{:.4f}.pkl'.format(t, auc_, f1_macro, f1_micro)))

    logger.info("Done!")
