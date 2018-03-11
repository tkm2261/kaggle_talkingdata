import os
import sys
import glob
import json
import random
import pickle
import gc
import numpy as np
import scipy.sparse as sp
import pandas as pd
import tensorflow as tf
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from dense import get_dense, LIST_DATA_COL, LIST_CONV_COL, LIST_FLOAT_COL, LIST_CAT_COL
import dask.dataframe as ddf
import dask.multiprocessing
from tqdm import tqdm
from multiprocessing.pool import Pool
from multiprocessing import Process, Queue


batch_size = 50000

np.random.seed(0)

from logging import getLogger

logger = getLogger(None)

DTYPE = None

try:
    param_file = sys.argv[1]
    with open(param_file) as f:
        model_params = json.loads(f.read())
except IndexError:
    param_file = None
    model_params = {'first_dences': [64, 32, 32, 8],  # [128, 32, 32],
                    'learning_rate': 0.001,
                    }


import gzip


def calc_batch_num(paths):
    cnt = 0
    for path in tqdm(paths):
        tmp = sum(1 for line in gzip.open(path)) - 1
        cnt += len(range(0, tmp, batch_size))
    return cnt


def data_generator(paths):
    for path in paths:
        logger.info(path)
        df = pd.read_csv(path, dtype=DTYPE).fillna(-1)

        for start in range(0, df.shape[0], batch_size):
            end = min(start + batch_size, df.shape[0])
            data = df.iloc[start:end]

            targets = np.array(data['click_id'].values)
            float_data = data[LIST_FLOAT_COL].values
            cat_data = [data[col].values for col in LIST_CAT_COL]
            x_batch = [float_data] + cat_data
            y_batch = targets

            yield x_batch, y_batch


from sklearn.metrics import roc_auc_score
DIR = './submit/'
import time


def producer(queue, paths, model):
    cnt = 0
    for x_batch, y_batch in data_generator(paths):
        cnt += 1
        click_ids = y_batch.tolist()
        pred = model.predict_on_batch(x_batch)[:, 0]
        queue.append((cnt, click_ids, pred))


def consumer(item):
    cnt = 0
    cnt, click_ids, pred = item
    sub = pd.DataFrame([click_ids, pred.tolist()]).T
    sub.columns = ['click_id', 'is_attributed']
    sub['click_id'] = sub['click_id'].astype(int)
    sub.to_csv(DIR + f'submit_{cnt}.csv', index=False)


def main():
    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')

    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.setLevel('INFO')
    logger.addHandler(handler)

    handler = FileHandler(DIR + 'pred.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info(f'file: {param_file}, params: {model_params}')
    paths = sorted(glob.glob('../data/dmt_test_raw/*.csv.gz'))
    model = get_dense(**model_params)
    model.load_weights(filepath='weights/best_weights.hdf5')
    model._make_predict_function()

    logger.info('model load end')

    queue = []
    producer(queue, paths, model)

    logger.info('pred load end')

    with Pool() as p:
        p.map(consumer, tqdm(queue), chunksize=20)


if __name__ == '__main__':
    main()
