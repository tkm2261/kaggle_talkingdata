import os
import sys
import glob
import json
import random
import pickle
import numpy as np
import scipy.sparse as sp
import pandas as pd
import tensorflow as tf
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lstm import get_lstm3, get_lstm_sin, MAX_SEQUENCE_LENGTH, LIST_DATA_COL, LIST_CONV_COL, get_lstm2, LIST_FLOAT_COL
import dask.dataframe as ddf
import dask.multiprocessing
from tqdm import tqdm
from multiprocessing.pool import Pool
batch_size = 1000

np.random.seed(0)

from logging import getLogger

logger = getLogger(None)

try:
    param_file = sys.argv[1]
    with open(param_file) as f:
        model_params = json.loads(f.read())
except IndexError:
    param_file = None
    model_params = {'first_dences': [64, 16, 16],  # [128, 32, 32],
                    'is_first_bn': False,
                    'is_last_bn': False,
                    'last_dences': [16, 8],  # [32, 16],
                    'learning_rate': 0.001,
                    'lstm_dropout': 0.15,
                    'lstm_recurrent_dropout': 0.15,
                    'lstm_size': 16  # 32
                    }


def read_csv(path):
    sv = 'cache/' + path.split('/')[-1].split('.')[0] + '.gz'
    if os.path.exists(sv):
        print(path)
        return pd.read_pickle(sv)

    df = pd.read_csv(path)

    for col in LIST_DATA_COL + ['list_target']:
        if col in LIST_CONV_COL:
            c = col.replace('avg_', '')
            postfix = col.split('_')[-1]
            map_d = pd.read_csv(f'../data/mst_{postfix}.csv', index_col=postfix).to_dict()[f'avg_{postfix}']
            df[col] = df[c].apply(lambda x: np.array([map_d.get(i, -1) for i in x], dtype=np.float32))
        else:
            df[col].fillna('[]', inplace=True)
            df[col] = df[col].apply(lambda x: np.array(json.loads(x), dtype=np.float32))
    df.to_pickle(sv, protocol=-1)
    print(path)
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


def data_generator(df):
    while True:
        df = df.take(np.random.permutation(len(df))).reset_index(drop=True)
        for start in range(0, df.shape[0], batch_size):
            end = min(start + batch_size, df.shape[0])
            data = df.iloc[start:end]

            list_click_ids = np.array(data['list_target'].values)
            data = data[LIST_FLOAT_COL].values
            batch_click_ids = []
            inputs = [[] for _ in range(data.shape[1])]

            results = [_proc_row((data[i], list_click_ids[i])) for i in range(len(list_click_ids))]

            logger.debug('proc end')
            for _ids, _data in results:
                batch_click_ids += _ids
                for k in range(data.shape[1]):
                    inputs[k] += _data[k]
            logger.debug('join end')
            inputs = [np.stack(inputs[k]) for k in range(data.shape[1])]

            yield inputs, batch_click_ids, list_click_ids


if __name__ == '__main__':

    from logging import StreamHandler, DEBUG, Formatter, FileHandler, NullHandler
    DIR = './'

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')

    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.setLevel('INFO')
    logger.addHandler(handler)

    handler = NullHandler()
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info(f'file: {param_file}, params: {model_params}')

    with Pool(processes=4) as pool:
        df = pd.concat(list(pool.map(read_csv, glob.glob('../data/dmt_train/*.csv.gz')[:1])),
                       ignore_index=True, copy=False)
        df.sort_values('ip', inplace=True)
        #df.to_pickle('train.pkl', protocol=-1)
    """
    df = pd.read_pickle('train.pkl')
    """

    logger.info('load end')

    ids_train = df.ip.values
    ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.1, random_state=42)

    train_df = df[df.ip.isin(ids_train_split)].reset_index(drop=True)
    valid_df = df[df.ip.isin(ids_valid_split)].reset_index(drop=True)
    del df
    logger.info('split end')

    valid = data_generator(valid_df)

    model = get_lstm_sin(**model_params)
    model.load_weights(filepath='weights/best_weights.hdf5')

    scores = []
    from sklearn.metrics import roc_auc_score
    preds = []
    labels = []
    for i in range(10):

        aaa = next(valid)
        with open('aaa.pkl', 'wb') as f:
            pickle.dump(aaa, f, -1)
        """
        with open('aaa.pkl', 'rb') as f:
            aaa = pickle.load(f)
        """
        x_batch, y_batch, ans = aaa

        pred = model.predict(x_batch, batch_size=1000, verbose=1)[:, 0]
        preds += pred.tolist()
        labels += y_batch
        sc = roc_auc_score(y_batch, pred)
        print(f'roc {i}', sc)
        scores.append(sc)
    print('mean', np.mean(scores))
    print('mean', roc_auc_score(labels, preds))
