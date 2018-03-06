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
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lstm import get_lstm, MAX_SEQUENCE_LENGTH, LIST_DATA_COL
import dask.dataframe as ddf
import dask.multiprocessing
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
batch_size = 1000


def read_csv(path):
    df = pd.read_csv(path)
    for col in LIST_DATA_COL + ['list_target']:
        df[col].fillna('[]', inplace=True)
        df[col] = df[col].apply(json.loads)
    print(path)
    return df


with ThreadPool(processes=4) as pool:
    df = pd.concat(list(pool.map(read_csv, glob.glob('../data/dmt_train/*.csv.gz'))),
                   ignore_index=True, copy=False)

df.to_pickle('train.pkl.gz', protocol=-1)
"""
df = pd.read_pickle('train.pkl.gz')
"""

ids_train = df.ip.values
ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.1, random_state=42)

train_df = df[df.ip.isin(ids_train_split)].reset_index(drop=True)
valid_df = df[df.ip.isin(ids_valid_split)].reset_index(drop=True)
del df


def pad(x, full=-1, dtype='int32'):
    ret = np.full(MAX_SEQUENCE_LENGTH, full, dtype=dtype)
    if len(x) > 1:
        end = random.randint(1, len(x))
    elif len(x) == 0:
        return ret
    else:
        end = len(x)
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
            data = data[LIST_DATA_COL].values

            inputs = []
            for i in range(data.shape[1]):
                inputs.append(np.array([pad(x) for x in data[:, i]], dtype='int32'))

            x_batch = inputs
            y_batch = np.array([pad(x, 0) for x in targets])
            y_batch = np.expand_dims(y_batch, axis=2)

            yield x_batch, y_batch


metric = 'val_auc'
callbacks = [EarlyStopping(monitor=metric,
                           patience=8,
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
             TensorBoard(log_dir='logs')]

if __name__ == '__main__':
    model = get_lstm()
    # model.load_weights(filepath='weights/best_weights.hdf5')

    epochs = 10000
    model.fit_generator(generator=data_generator(train_df),
                        steps_per_epoch=np.ceil(float(len(ids_train_split)) / float(batch_size)),
                        epochs=epochs,
                        verbose=2,
                        callbacks=callbacks,
                        validation_data=data_generator(valid_df),
                        validation_steps=np.ceil(float(len(ids_valid_split)) / float(batch_size)))
