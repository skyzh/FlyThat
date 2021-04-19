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


def main(args):
    parser = argparse.ArgumentParser(description='Generate PyTorch Dataset')
    parser.add_argument('data_path', type=str, nargs='?', help='Path of unzip data',
                        default=str(Path() / 'data'))
    args = parser.parse_args(args)
    data_path = Path(args.data_path)
    logger.info(f"using data path '{data_path}'")

    device = utils.get_device()
    logger.info(f'Using {device} device')
    model = NeuralNetwork()
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

    for t in range(epochs):
        logger.info(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)
    logger.info("Done!")
