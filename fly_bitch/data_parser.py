import pandas as pd
import argparse
from pathlib import Path
from loguru import logger
from ast import literal_eval


def parse_tuple_like(data):
    return tuple(data.strip('()').split(','))


def get_train_data(csv_path):
    """Generate train data in pandas Dataframe format

    Args:
        csv_path (pathlib.Path): Path of CSV file

    Returns:
        pandas.DataFrame: a dataframe
    """

    df = pd.read_csv(csv_path)
    df['labels'] = df['labels'].apply(lambda x: tuple(map(int, x.split(','))))
    df['imgs'] = df['imgs'].apply(parse_tuple_like)
    return df


def main(args):
    parser = argparse.ArgumentParser(description='Parse Fly data')
    parser.add_argument('data_path', type=str, nargs='?', help='Path of unzip data',
                        default=str(Path() / 'data' / 'train.csv'))
    args = parser.parse_args(args)
    data_path = Path(args.data_path)

    logger.info(f"using data path '{data_path}'")

    df = get_train_data(data_path)
    logger.info("items from `get_train_data`:")
    print(df)

    logger.info(f'image number stats:')
    print(df['imgs'].apply(len).describe())
