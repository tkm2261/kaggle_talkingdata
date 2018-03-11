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
from lstm import get_lstm_sin, MAX_SEQUENCE_LENGTH, LIST_DATA_COL, LIST_CONV_COL, LIST_FLOAT_COL
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


def rand_end(x):
    if len(x) > MAX_SEQUENCE_LENGTH:
        end = random.randint(MAX_SEQUENCE_LENGTH, len(x))
    else:
        end = len(x)
    return end


def pad(x, end, full=-1, dtype='int32'):
    ret = np.full(MAX_SEQUENCE_LENGTH, full, dtype=dtype)
    start = max(0, end - MAX_SEQUENCE_LENGTH)
    ret[-(end - start):] = x[start:end]
    return ret


def data_generator(df):
    while True:
        df = df.take(np.random.permutation(len(df))).reset_index(drop=True)
        for start in range(0, df.shape[0], batch_size):
            end = min(start + batch_size, df.shape[0])
            data = df.iloc[start:end]

            targets = np.array(data['list_target'].values)
            data = data[LIST_FLOAT_COL].values
            rand_ends = [rand_end(x) for x in targets]

            inputs = [np.array([pad(x, rand_ends[i]) for i, x in enumerate(data[:, i])], dtype=df[LIST_DATA_COL[i]].dtype)
                      for i in range(data.shape[1])]

            x_batch = inputs
            y_batch = np.array([x[rand_ends[i] - 1] for i, x in enumerate(targets)])
            #y_batch = np.expand_dims(y_batch, axis=2)

            yield x_batch, y_batch


class LoggingCallback(Callback):
    """Callback that logs message at end of epoch.
    """

    def on_epoch_end(self, epoch, logs={}):
        msg = "Epoch: %i, %s" % (epoch, ", ".join("%s: %f" % (k, logs[k]) for k in sorted(logs)))
        logger.info(msg)


metric = 'val_auc'
callbacks = [EarlyStopping(monitor=metric,
                           patience=3,
                           verbose=1,
                           min_delta=1e-4,
                           mode='max'),
             ReduceLROnPlateau(monitor=metric,
                               factor=0.1,
                               patience=4,
                               verbose=1,
                               epsilon=1e-4,
                               mode='max'),
             ModelCheckpoint(monitor=metric,
                             filepath='weights/best_weights.hdf5',
                             save_best_only=True,
                             save_weights_only=True,
                             mode='max'),
             TensorBoard(log_dir='logs'),
             LoggingCallback()
             ]

if __name__ == '__main__':

    from logging import StreamHandler, DEBUG, Formatter, FileHandler
    DIR = './'

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')

    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.setLevel('INFO')
    logger.addHandler(handler)

    handler = FileHandler(DIR + 'train.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info(f'file: {param_file}, params: {model_params}')
    with Pool(processes=4) as pool:
        df = pd.concat(list(pool.map(read_csv, glob.glob('../data/dmt_train/*.csv.gz'))),
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

    model = get_lstm_sin(**model_params)
    # model.load_weights(filepath='weights/best_weights.hdf5')

    epochs = 10000
    model.fit_generator(generator=data_generator(train_df),
                        steps_per_epoch=np.ceil(float(len(ids_train_split)) / float(batch_size)),
                        epochs=epochs,
                        verbose=2,
                        callbacks=callbacks,
                        validation_data=data_generator(valid_df),
                        validation_steps=np.ceil(float(len(ids_valid_split)) / float(batch_size)))

    """
    valid = data_generator(valid_df)
    scores = []
    from sklearn.metrics import roc_auc_score
    preds = []
    labels = []
    for i in range(10):
        x_batch, y_batch = next(valid)
        pred = model.predict(x_batch, batch_size=1000, verbose=1)[:, -1, 0].flatten().tolist()
        preds += pred
        labels += y_batch[:, -1, 0].flatten().tolist()
        sc = roc_auc_score(y_batch[:, -1, 0].flatten(), pred)
        print(f'roc {i}', sc)
        scores.append(sc)
    print('mean', np.mean(scores))
    print('mean', roc_auc_score(labels, preds))
    """
