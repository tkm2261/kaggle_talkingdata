import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
from logging import getLogger
from multiprocessing import Pool
logger = getLogger(None)

TRAIN_DATA_DIR = './data/95/numerai_training_data.csv'
TEST_DATA_DIR = './data/95/numerai_tournament_data.csv'
"""
DTYPE = {'click_id': np.float32,  # np.int64,
         'ip': np.float32,
         'app': np.float32,
         #'app_1': np.float32,
         #'app_2': np.float32,
         #'app_3': np.float32,
         #'app_4': np.float32,
         'device': np.float32,
         #'device_1': np.float32,
         #'device_2': np.float32,
         #'device_3': np.float32,
         #'device_4': np.float32,
         'os': np.float32,
         #'os_1': np.float32,
         #'os_2': np.float32,
         #'os_3': np.float32,
         #'os_4': np.float32,
         'channel': np.float32,
         #'channel_1': np.float32,
         #'channel_2': np.float32,
         #'channel_3': np.float32,
         #'channel_4': np.float32,
         'is_attributed': np.float32,
         #'is_attributed_1': np.float32,
         #'is_attributed_2': np.float32,
         #'is_attributed_3': np.float32,
         #'is_attributed_4': np.float32,
         #'is_attributed_5': np.float32,
         'hour': np.float32,
         #'hour_1': np.float32,
         #'hour_2': np.float32,
         #'hour_3': np.float32,
         #'hour_4': np.float32,
         'avg_app': np.float32,
         #'avg_app_1': np.float32,
         #'avg_app_2': np.float32,
         #'avg_app_3': np.float32,
         #'avg_app_4': np.float32,
         'avg_device': np.float32,
         #'avg_device_1': np.float32,
         #'avg_device_2': np.float32,
         #'avg_device_3': np.float32,
         #'avg_device_4': np.float32,
         'avg_os': np.float32,
         #'avg_os_1': np.float32,
         #'avg_os_2': np.float32,
         #'avg_os_3': np.float32,
         #'avg_os_4': np.float32,
         'avg_channel': np.float32,
         #'avg_channel_1': np.float32,
         #'avg_channel_2': np.float32,
         #'avg_channel_3': np.float32,
         #'avg_channel_4': np.float32,
         'avg_day': np.float32,
         #'avg_day_1': np.float32,
         #'avg_day_2': np.float32,
         #'avg_day_3': np.float32,
         #'avg_day_4': np.float32,
         'avg_hour': np.float32,
         #'avg_hour_1': np.float32,
         #'avg_hour_2': np.float32,
         #'avg_hour_3': np.float32,
         #'avg_hour_4': np.float32
         }
"""
DTYPE = {'click_id': np.float32,
         'ip': np.float32,
         'app': np.float32,
         'app_1': np.float32,
         'app_2': np.float32,
         'app_3': np.float32,
         'app_4': np.float32,
         'device': np.float32,
         'device_1': np.float32,
         'device_2': np.float32,
         'device_3': np.float32,
         'device_4': np.float32,
         'os': np.float32,
         'os_1': np.float32,
         'os_2': np.float32,
         'os_3': np.float32,
         'os_4': np.float32,
         'channel': np.float32,
         'channel_1': np.float32,
         'channel_2': np.float32,
         'channel_3': np.float32,
         'channel_4': np.float32,
         'is_attributed': np.float32,
         'is_attributed_1': np.float32,
         'is_attributed_2': np.float32,
         'is_attributed_3': np.float32,
         'is_attributed_4': np.float32,
         'is_attributed_5': np.float32,
         'hour': np.float32,
         'hour_1': np.float32,
         'hour_2': np.float32,
         'hour_3': np.float32,
         'hour_4': np.float32,
         'avg_app': np.float32,
         'avg_app_1': np.float32,
         'avg_app_2': np.float32,
         'avg_app_3': np.float32,
         'avg_app_4': np.float32,
         'avg_device': np.float32,
         'avg_device_1': np.float32,
         'avg_device_2': np.float32,
         'avg_device_3': np.float32,
         'avg_device_4': np.float32,
         'avg_os': np.float32,
         'avg_os_1': np.float32,
         'avg_os_2': np.float32,
         'avg_os_3': np.float32,
         'avg_os_4': np.float32,
         'avg_channel': np.float32,
         'avg_channel_1': np.float32,
         'avg_channel_2': np.float32,
         'avg_channel_3': np.float32,
         'avg_channel_4': np.float32,
         'avg_day': np.float32,
         'avg_day_1': np.float32,
         'avg_day_2': np.float32,
         'avg_day_3': np.float32,
         'avg_day_4': np.float32,
         'avg_hour': np.float32,
         'avg_hour_1': np.float32,
         'avg_hour_2': np.float32,
         'avg_hour_3': np.float32,
         'avg_hour_4': np.float32
         }

import copy


def read_csv(path):
    dtype = copy.deepcopy(DTYPE)
    if 'test' in path:
        dtype['click_id'] = np.int64

    df = pd.read_csv(path, dtype=dtype, usecols=list(dtype.keys()))
    return df


def load_train_data():
    paths = sorted(glob.glob('../data/dmt_train_lag/*.csv.gz'))
    with Pool() as p:
        df = pd.concat(p.map(read_csv, paths), ignore_index=True, axis=0, copy=False)
    logger.info('data size {}'.format(df.shape))
    return df


def load_test_data():
    paths = sorted(glob.glob('../data/dmt_test_lag/*.csv.gz'))
    with Pool() as p:
        df = pd.concat(p.map(read_csv, paths), ignore_index=True, axis=0, copy=False)
    logger.info('data size {}'.format(df.shape))
    return df


if __name__ == '__main__':
    load_train_data()
    load_test_data()
