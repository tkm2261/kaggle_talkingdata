import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
from logging import getLogger
from multiprocessing import Pool
logger = getLogger(None)

import copy

LIST_COL = ['span', 'click_id', 'is_attributed', 'channel', 'os', 'app', 'hour', 'rt_dayhouripos', 'cnt_dayiphourapp', 'diff_all', 'rt_dayhouripdevice',
            'diff_dayhouripos', 'cnt_dayhouripos', 'diff_dayiphourapp', 'rt_dayhouripapp', 'cnt_dayhouripdevice', 'cnt_dayhouripapp', 'diff_dayhouripapp']


def read_csv(path):
    # dtype = copy.deepcopy(DTYPE)
    # if 'test' in path:
    #    dtype['click_id'] = np.int64

    # df = pd.read_csv(path, dtype=dtype, usecols=list(dtype.keys()))
    df = pd.read_csv(path)  # , usecols=LIST_COL)  # [list(dtype.keys())]
    df.drop(['ip', 'day', 'cnt_ch_day'] + ['dist_channel', 'dist_os', 'dist_app',
                                           'dist_device'] + ['cnt_dayspanipapp', 'cnt_dayspanipos', 'cnt_dayspanipdevice']
            + ['cnt_dayhourapp', 'cnt_dayhouros', 'cnt_dayhourdevice'], axis=1, inplace=True, errors='ignore')

    for col in df:
        if col == 'click_id' and 'test' in path:
            df[col] = df[col].astype(np.int64)
        else:
            df[col] = df[col].astype(np.float32)
    df = df[df['span'] > 0]
    df.drop(['span'], axis=1, inplace=True)
    return df


def load_train_data():
    # + sorted(glob.glob('../data/dmt_prev_cnt3/*.csv.gz'))
    paths = sorted(glob.glob('../data/dmt_train_cnt3/*.csv.gz'))
    with Pool() as p:
        df = pd.concat(p.map(read_csv, paths), ignore_index=True, axis=0, copy=False)
    logger.info('data size {}'.format(df.shape))
    return df


def load_valid_data():
    paths = sorted(glob.glob('../data/dmt_valid_cnt3/*.csv.gz'))
    with Pool() as p:
        df = pd.concat(p.map(read_csv, paths), ignore_index=True, axis=0, copy=False)
    logger.info('data size {}'.format(df.shape))
    return df


def load_test_data():
    paths = sorted(glob.glob('../data/dmt_test_cnt3/*.csv.gz'))
    with Pool() as p:
        df = pd.concat(p.map(read_csv, paths), ignore_index=True, axis=0, copy=False)
    logger.info('data size {}'.format(df.shape))
    return df


def load_all_data():
    paths = sorted(glob.glob('../data/dmt_train_cnt3/*.csv.gz')) + \
        sorted(glob.glob('../data/dmt_valid_cnt3/*.csv.gz'))
    # sorted(glob.glob('../data/dmt_prev_cnt3/*.csv.gz'))

    with Pool() as p:
        df = pd.concat(p.map(read_csv, paths), ignore_index=True, axis=0, copy=False)
    logger.info('data size {}'.format(df.shape))
    return df


if __name__ == '__main__':
    load_train_data()
    load_test_data()
