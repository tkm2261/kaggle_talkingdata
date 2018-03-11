import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
from logging import getLogger
from multiprocessing import Pool
logger = getLogger(None)

TRAIN_DATA_DIR = './data/95/numerai_training_data.csv'
TEST_DATA_DIR = './data/95/numerai_tournament_data.csv'


def read_csv(path):
    df = pd.read_csv(path)
    return df


def load_train_data():
    paths = sorted(glob.glob('../data/dmt_train_lag/*.csv.gz'))
    with Pool() as p:
        df = pd.concat(p.map(read_csv, paths, chunksize=12), ignore_index=True, axis=0, copy=False)
    logger.info('data size {}'.format(df.shape))
    return df


def load_test_data():
    paths = sorted(glob.glob('../data/dmt_test_lag/*.csv.gz'))
    with Pool() as p:
        df = pd.concat(p.map(read_csv, paths, chunksize=12), ignore_index=True, axis=0, copy=False)
    logger.info('data size {}'.format(df.shape))
    return df


if __name__ == '__main__':
    load_train_data()
    load_test_data()
