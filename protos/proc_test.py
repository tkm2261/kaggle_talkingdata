import os
import sys
import glob
import json
import gc
import random
import pickle
import numpy as np
import scipy.sparse as sp
import pandas as pd
import tensorflow as tf
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lstm import get_lstm, MAX_SEQUENCE_LENGTH, LIST_DATA_COL, LIST_CONV_COL, get_lstm2
import dask.dataframe as ddf
import dask.multiprocessing
from tqdm import tqdm
from multiprocessing.pool import Pool
batch_size = 100

from logging import getLogger

logger = getLogger(None)


def read_csv(path):
    sv = 'cache/test/' + path.split('/')[-1].split('.')[0] + '.pkl'
    logger.info(path)

    df = pd.read_csv(path)
    col = 'list_click_id'
    df[col].fillna('[]', inplace=True)
    df[col] = df[col].apply(lambda x: np.array(json.loads(x), dtype=np.int64))

    for col in tqdm(LIST_DATA_COL):
        if col in LIST_CONV_COL:
            c = col.replace('avg_', '')
            postfix = col.split('_')[-1]
            map_d = pd.read_csv(f'../data/mst_{postfix}.csv', index_col=postfix).to_dict()[f'avg_{postfix}']
            df[col] = df[c].apply(lambda x: np.array([map_d.get(i, -1) for i in x], dtype=np.float32))
        else:
            df[col].fillna('[]', inplace=True)
            df[col] = df[col].apply(lambda x: np.array(json.loads(x), dtype=np.float32))
    df.to_pickle(sv, protocol=-1)
    return df


def pad(x, end, full=-1, dtype='int32'):
    ret = np.full(MAX_SEQUENCE_LENGTH, full, dtype=dtype)
    start = max(0, end - MAX_SEQUENCE_LENGTH)
    ret[-(end - start):] = x[start:end]
    return ret


def _proc_row(args):
    row, click_ids = args
    batch_click_ids = []
    inputs = [[] for _ in range(row.shape[0])]

    for j, click_id in enumerate(click_ids):
        if click_id == -1:
            continue
        batch_click_ids.append(click_id)
        for k in range(row.shape[0]):
            inputs[k].append(pad(row[k], j + 1))
    return batch_click_ids, inputs


if __name__ == '__main__':

    from logging import StreamHandler, DEBUG, Formatter, FileHandler, NullHandler
    DIR = './submit/'

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')

    handler = StreamHandler()
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    handler = NullHandler()
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)
    logger.info('start')

    cnt = 0
    itr = 0
    with Pool() as p:
        p.map(read_csv, glob.glob('../data/dmt_test/*.csv.gz'))
