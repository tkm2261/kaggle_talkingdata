import re
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from multiprocessing import Pool
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, cross_val_predict
# import xgboost as xgb
from lightgbm.sklearn import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, f1_score, log_loss
import gc
from logging import getLogger
logger = getLogger(None)

from tqdm import tqdm

from load_data import load_train_data, load_test_data, load_valid_data, load_all_data
import sys
DIR = 'result_tmp/'  # sys.argv[1]  # 'result_1008_rate001/'
print(DIR)

from numba import jit
CAT_FEAT = ['app', 'os']

LIST_DROP_COL = ['channel', 'cnt_os', 'cnt_dayhouripappos', 'cnt_ch', 'cnt_dayhouripos', 'cnt_dayhouripchannel', 'cnt_os_r', 'cnt_nochannel', 'cnt_dayhouripapp', 'cnt_ch_r', 'cnt_dayhouripappchannel',
                 'cnt_dayhouripappdevice', 'cnt_ip_r', 'cnt_dayiphourapp', 'cnt_dayhouripchanneldevice', 'cnt_dayhouripchannelos', 'cnt_app', 'cnt_ch_app', 'cnt_app_r', 'cnt_dayhouripdevice', 'cnt_ch_app_r', 'cnt_dayhouriosdevice', 'cnt_ip']


def consist_score(label, pred):
    idx = label == 1
    score = ((- np.log(pred[idx])) < - np.log(0.5)).sum()

    idx = label == 0
    score += ((- np.log(1 - pred[idx])) < - np.log(0.5)).sum()
    return score / label.shape[0]


def cst_metric_xgb(pred, dtrain):
    label = dtrain.get_label().astype(np.int)
    preds = pred.reshape((21, -1)).T
    preds = np.array([np.argmax(x) for x in preds], dtype=np.int)
    sc = log_loss(label, preds)
    return 'qwk', sc, True


def dummy(pred, dtrain):
    return 'dummy', pred, True


def callback(data):
    if (data.iteration + 1) % 10 != 0:
        return

    clf = data.model
    trn_data = clf.train_set
    val_data = clf.valid_sets[0]
    preds = [ele[2] for ele in clf.eval_valid(dummy) if ele[1] == 'dummy'][0]
    preds = preds.reshape((21, -1)).T
    preds = np.array([np.argmax(x) for x in preds], dtype=np.int)
    labels = val_data.get_label().astype(np.int)
    sc = log_loss(labels, preds)
    sc2 = roc_auc_score(labels, preds)
    logger.info('cal [{}] {} {}'.format(data.iteration + 1, sc, sc2))


def train():
    df = load_train_data()  # .sample(10000000, random_state=42).reset_index(drop=True)

    logger.info('train data size {}'.format(df.shape))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=871)

    train, test = next(cv.split(df, df.is_attributed))

    x_train = df.drop(['is_attributed', 'click_id'], axis=1).astype(np.float32)  # .loc[train].reset_index(drop=True)
    y_train = df.is_attributed.astype(int)  # .values[train]

    df = load_valid_data()  # .sample(x_train.shape[0], random_state=42).reset_index(drop=True)
    logger.info('valid data size {}'.format(df.shape))
    x_valid = df.drop(['is_attributed', 'click_id'], axis=1).astype(np.float32)  # .loc[test].reset_index(drop=True)
    y_valid = df.is_attributed.astype(int)  # .values[test]

    del df
    gc.collect()
    usecols = x_train.columns.values
    with open(DIR + 'usecols.pkl', 'wb') as f:
        pickle.dump(usecols, f, -1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=871)
    # {'colsample_bytree': 0.9, 'learning_rate': 0.1, 'max_bin': 255, 'max_depth': -1, 'metric': 'auc', 'min_child_weight': 20, 'min_split_gain': 0, 'num_leaves': 127, 'objective': 'binary', 'reg_alpha': 0, 'scale_pos_weight': 1, 'seed': 114, 'subsample': 1.0, 'subsample_freq': 1, 'verbose': -1}
    all_params = {'min_child_weight': [20],
                  'subsample': [1],
                  'subsample_freq': [1],
                  'seed': [114],
                  'colsample_bytree': [0.9],
                  'learning_rate': [0.1],
                  'max_depth': [-1],
                  'min_split_gain': [0],
                  'reg_alpha': [0],
                  'max_bin': [63],
                  'num_leaves': [127],
                  'objective': ['binary'],
                  'metric': ['auc'],
                  'scale_pos_weight': [1],
                  'verbose': [-1],
                  'device': ['gpu'],
                  'drop': list(range(1, len(LIST_DROP_COL)))
                  }
    use_score = 0
    min_score = (100, 100, 100)
    drop_cols = LIST_DROP_COL[:1]
    import copy
    for params in tqdm(list(ParameterGrid(all_params))):
        cnt = -1
        list_score = []
        list_score2 = []
        list_best_iter = []
        all_pred = np.zeros(y_train.shape[0])
        if 1:
            cnt += 1
            trn_x = x_train.copy()
            val_x = x_valid.copy()
            trn_y = y_train
            val_y = y_valid

            _params = copy.deepcopy(params)
            drop_idx = _params.pop('drop')
            drop_col = drop_cols + [LIST_DROP_COL[drop_idx]]
            params['drop'] = drop_col

            trn_x.drop(drop_col, axis=1, inplace=True)
            val_x.drop(drop_col, axis=1, inplace=True)
            cat_feat = CAT_FEAT
            cols = trn_x.columns.values.tolist()
            train_data = lgb.Dataset(trn_x.values.astype(np.float32), label=trn_y,
                                     categorical_feature=cat_feat, feature_name=cols)
            test_data = lgb.Dataset(val_x.values.astype(np.float32), label=val_y,
                                    categorical_feature=cat_feat, feature_name=cols)
            del trn_x
            gc.collect()

            clf = lgb.train(_params,
                            train_data,
                            10000,  # params['n_estimators'],
                            early_stopping_rounds=30,
                            valid_sets=[test_data],
                            # feval=cst_metric_xgb,
                            # callbacks=[callback],
                            verbose_eval=10
                            )
            pred = clf.predict(val_x.values.astype(np.float32))

            # all_pred[test] = pred

            _score2 = log_loss(val_y, pred)
            _score = - roc_auc_score(val_y, pred)
            logger.info(f'drop: {drop_col}')
            logger.info('   _score: %s' % _score)
            logger.info('   _score2: %s' % _score2)

            list_score.append(_score)
            list_score2.append(_score2)
        score = (np.mean(list_score), np.min(list_score), np.max(list_score))
        score2 = (np.mean(list_score2), np.min(list_score2), np.max(list_score2))
        if min_score[use_score] > score[use_score]:
            min_score = score
            min_params = params
            drop_cols = drop_col
        logger.info('best score: {} {}'.format(min_score[use_score], min_score))
        logger.info('best params: {}'.format(min_params))


if __name__ == '__main__':

    from logging import StreamHandler, DEBUG, Formatter, FileHandler

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

    train()
