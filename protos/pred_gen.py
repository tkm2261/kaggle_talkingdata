import os
import sys
import glob
import json
import gc
import time
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
from tqdm import tqdm
from multiprocessing.pool import Pool
from multiprocessing import Process, Queue
batch_size = 100

from logging import getLogger

logger = getLogger(None)

FILE_DIR = os.path.dirname(os.path.abspath(__file__))


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


def producer(queue):
    itr = 0
    for path in sorted(glob.glob(os.path.join(FILE_DIR, 'cache/test/*.pkl'))):
        logger.debug('load start')
        logger.info(path)
        df = pd.read_pickle(path)
        logger.debug('load end')

        for start in range(0, df.shape[0], batch_size):
            itr += 1
            if os.path.exists(DIR + f'submit_{itr}.csv'):
                logger.info(f'pred: {itr}')
                continue

            end = min(start + batch_size, df.shape[0])
            data = df.iloc[start:end]

            list_click_ids = np.array(data['list_click_id'].values)
            data = data[LIST_DATA_COL].values
            batch_click_ids = []
            inputs = [[] for _ in range(data.shape[1])]

            results = [_proc_row((data[i], list_click_ids[i])) for i in range(len(list_click_ids))]

            # with Pool(processes=4) as p:
            # results = map(_proc_row,
            #              [(data[i], list_click_ids[i]) for i in range(len(list_click_ids))],
            #              # chunksize=100
            #              )
            logger.debug('proc end')
            for _ids, _data in results:
                batch_click_ids += _ids
                for k in range(data.shape[1]):
                    inputs[k] += _data[k]
            logger.debug('join end')
            inputs = [np.stack(inputs[k]) for k in range(data.shape[1])]
            queue.put((itr, batch_click_ids, inputs))
            while queue.qsize() > 5:
                time.sleep(1)

    queue.put(None)


def consumer(model, queue):
    cnt = 0
    while True:
        # wait for an item from the producer
        item = queue.get()
        if item is None:
            # the producer emits None to indicate that it is done
            break

        itr, batch_click_ids, inputs = item

        cnt += inputs[0].shape[0]
        logger.info(f'pred: {itr}, prog: {cnt / 18790469: .3}, raw: {(cnt, 18790469)}')

        logger.debug('pred start')
        pred = model.predict(inputs, batch_size=1000, verbose=1)[:, -1, 0]
        logger.debug('pred end')

        sub = pd.DataFrame([batch_click_ids, pred.tolist()]).T
        sub.columns = ['click_id', 'is_attributed']
        sub['click_id'] = sub['click_id'].astype(int)
        sub.to_csv(DIR + f'submit_{itr}.csv', index=False)


if __name__ == '__main__':

    from logging import StreamHandler, DEBUG, Formatter, FileHandler, INFO
    DIR = os.path.join(FILE_DIR, 'submit/')

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')

    handler = StreamHandler()
    handler.setLevel(INFO)
    handler.setFormatter(log_fmt)
    logger.setLevel(INFO)
    logger.addHandler(handler)

    handler = FileHandler(os.path.join(DIR + 'pred.py.log'), 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)
    logger.info('start')
    """
    with Pool(processes=4) as pool:
        df = pd.concat(list(pool.map(read_csv, glob.glob('../data/dmt_test/*.csv.gz'))),
                       ignore_index=True, copy=False)
    """
    model_params = {'first_dences': [64, 16, 16],  # [128, 32, 32],
                    'is_first_bn': False,
                    'is_last_bn': False,
                    'last_dences': [16, 8],  # [32, 16],
                    'learning_rate': 0.001,
                    'lstm_dropout': 0.15,
                    'lstm_recurrent_dropout': 0.15,
                    'lstm_size': 16  # 32
                    }

    model = get_lstm2(**model_params)
    model.load_weights(filepath=os.path.join(FILE_DIR, 'weights/best_weights_0310_sub1.hdf5'))
    model._make_predict_function()
    logger.info('model load end')

    queue = Queue()
    workers = [Process(target=producer, args=(queue,)) for _ in range(1)]
    [worker.start() for worker in workers]

    consumer(model, queue)
