import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
from logging import getLogger
from multiprocessing import Pool
logger = getLogger(None)

import copy
DROP = ['diff_dayhouripchannelos', 'diff_dayhouripchannel', 'diff_dayhouripapp', 'diff_dayhouripchanneldevice', 'diff_dayhouriosdevice',
        'diff_dayhouripappdevice', 'diff_dayhouripdevice', 'diff_noos', 'diff_noapp', 'diff_nodevice', 'diff_dayiphourapp', 'diff_nochannel']
LIST_COL = ['span', 'click_id', 'is_attributed', 'channel', 'os', 'app', 'hour', 'rt_dayhouripos', 'cnt_dayiphourapp', 'diff_all', 'rt_dayhouripdevice',
            'diff_dayhouripos', 'cnt_dayhouripos', 'diff_dayiphourapp', 'rt_dayhouripapp', 'cnt_dayhouripdevice', 'cnt_dayhouripapp', 'diff_dayhouripapp']

LIST_ROWS = ['row_ip', 'row_os', 'row_app', 'row_channel', 'row_device', 'row_os_app', 'row_os_channel', 'row_os_device',
             'row_app_channel', 'row_app_device', 'row_channel_device', 'row_no_os', 'row_no_app', 'row_no_channel', 'row_no_device', 'row_all']


def read_csv(path):
    # dtype = copy.deepcopy(DTYPE)
    # if 'test' in path:
    #    dtype['click_id'] = np.int64

    # df = pd.read_csv(path, dtype=dtype, usecols=list(dtype.keys()))
    df = pd.read_csv(path)  # , usecols=LIST_COL)  # [list(dtype.keys())]
    df.drop(['ip', 'day', 'cnt_ch_day', 'hour'] + ['channel', 'cnt_all'] + DROP, axis=1, inplace=True, errors='ignore')

    for col in LIST_ROWS:
        df[col] = df[col] / (df[col] + df[col + '_r'])
        df.drop(col + '_r', axis=1, inplace=True)

    for col in df:
        if col == 'click_id' and 'test' in path:
            df[col] = df[col].astype(np.int64)
        else:
            df[col] = df[col].astype(np.float32)
    # df = df[(df['span'] > 0)]  # | (df['is_attributed'] == 1)]
    df.drop(['span'], axis=1, inplace=True)
    return df


def load_train_data():
    #
    #
    paths = sorted(glob.glob('../data/dmt_train_0407/*.csv.gz')) + sorted(glob.glob('../data/dmt_prev_0407/*.csv.gz'))
    with Pool() as p:
        df = pd.concat(p.map(read_csv, paths), ignore_index=True, axis=0, copy=False)
    logger.info('data size {}'.format(df.shape))
    return df


def load_valid_data():
    paths = sorted(glob.glob('../data/dmt_valid_0407/*.csv.gz'))
    with Pool() as p:
        df = pd.concat(p.map(read_csv, paths), ignore_index=True, axis=0, copy=False)
    logger.info('data size {}'.format(df.shape))
    return df


def load_test_data():
    paths = sorted(glob.glob('../data/dmt_test_0407/*.csv.gz'))
    with Pool() as p:
        df = pd.concat(p.map(read_csv, paths), ignore_index=True, axis=0, copy=False)
    logger.info('data size {}'.format(df.shape))
    return df


def load_all_data():
    paths = sorted(glob.glob('../data/dmt_train_0407/*.csv.gz')) + \
        sorted(glob.glob('../data/dmt_valid_0407/*.csv.gz')) + sorted(glob.glob('../data/dmt_prev_0407/*.csv.gz'))

    with Pool() as p:
        df = pd.concat(p.map(read_csv, paths), ignore_index=True, axis=0, copy=False)
    logger.info('data size {}'.format(df.shape))
    return df


if __name__ == '__main__':
    load_train_data()
    load_test_data()
