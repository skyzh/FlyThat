from fly_bitch.dataset import DrosophilaTestImageDataset
from pathlib import Path
import argparse
from loguru import logger
import re
import os
from . import utils
import torch
from .feature_extractor import feature_transformer
from torch.utils.data import Dataset, DataLoader
from .model import NeuralNetwork
from .train import result_threshold
import numpy as np
import pandas as pd


def parse_file_name(filename):
    pattern = '^model_(.*?)_(.*?)_(.*?)_(.*?)\\.pkl$'
    result = re.match(pattern, filename)
    return int(result[1]), float(result[2]), float(result[3]), float(result[4])


def sort_by_auc(data):
    val, _ = data
    _, auc, _, _ = val
    return auc


def log_models(data):
    from prettytable import PrettyTable
    x = PrettyTable()
    x.field_names = ["Epoch", "AUC", "F1 Macro", "F1 Micro"]
    for row, _ in data:
        x.add_row(row)
    print(x)


def gen_data(args, model):
    (epoch, _, _, _), path = model
    logger.info(f"Using model from #{epoch}")
    device = utils.get_device()
    logger.info(f'Using {device} device')
    model = NeuralNetwork()
    model = model.to(device)
    model.load_state_dict(torch.load(path).state_dict())
    dataset = DrosophilaTestImageDataset(
        Path(args.data), feature_transformer(), args.partial)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=False)
    all_names = []
    all_outputs = []
    all_scores = []

    with torch.no_grad():
        size = len(dataloader.dataset)
        processed = 0
        for names, data in dataloader:
            processed += len(data)
            logger.info(f"Processing [{processed}/{size}]")
            data = data.to(device)
            outputs = model(data)
            outputs = outputs.cpu().numpy()
            all_names.extend(names)
            for output in outputs:
                all_scores.append(",".join(map(str, output)))
            for output in result_threshold(outputs):
                all_outputs.append(
                    ",".join(map(str, np.where(output == 1)[0])))
            df = pd.DataFrame(
                {'gene_stage': all_names, 'predict': all_outputs, 'score': all_scores})


def main(argv):
    DEFAULT_MODEL_PATH = str(Path() / 'runs/model')
    parser = argparse.ArgumentParser(description='Evaluate Model')
    parser.add_argument('--model', type=str, nargs='?', help='Model checkpoint path',
                        default=DEFAULT_MODEL_PATH)
    parser.add_argument('--data', type=str, nargs='?', help='Path of unzip data',
                        default=str(Path() / 'data'))
    parser.add_argument('--batch', type=int, nargs='?', help='Batch size',
                        default=64)
    parser.add_argument('--partial', action='store_true',
                        help='Only use part of the data', default=False)

    args = parser.parse_args(argv)
    model_folder = args.model or DEFAULT_MODEL_PATH

    logger.info(f"Using model folder {model_folder}")
    models = []
    for entry in os.scandir(model_folder):
        if entry.is_file():
            if entry.name.startswith('model'):
                models.append((parse_file_name(entry.name), entry.path))
    logger.info(f"Found {len(models)} models")
    models.sort(key=sort_by_auc, reverse=True)
    log_models(models)
    gen_data(args, models[0])
