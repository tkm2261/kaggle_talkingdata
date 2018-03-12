import re
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from multiprocessing import Pool
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, cross_val_predict
#import xgboost as xgb
from lightgbm.sklearn import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, f1_score, log_loss
import gc
from logging import getLogger
logger = getLogger(None)

from tqdm import tqdm

from load_data import load_train_data, load_test_data
import sys
DIR = 'result_tmp/'  # sys.argv[1]  # 'result_1008_rate001/'
print(DIR)

from numba import jit


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

    #df = load_train_data()
    #df.to_pickle('train.pkl', protocol=-1)

    df = pd.read_pickle('train.pkl')  # .tail(50000000).reset_index(drop=True)
    pos = df[df.is_attributed == 1]
    neg = df[df.is_attributed == 0]
    n_pos = pos.shape[0]
    n_neg = neg.shape[0]
    factor = 10
    df = pd.concat([pos, neg.sample(n_pos * factor, random_state=42)], axis=0, ignore_index=True, copy=False)
    scale_pos_weight = n_pos / n_neg * factor

    df.to_pickle('train_sampling.pkl', protocol=-1)

    del pos
    del neg
    gc.collect()
    logger.info('data size {}'.format(df.shape))
    logger.info(f'pos: {n_pos}, neg: {n_neg}, rate: {scale_pos_weight}')

    x_train = df.drop(['is_attributed', 'click_id'], axis=1).astype(np.float32)
    y_train = df.is_attributed.astype(int)
    usecols = x_train.columns.values
    with open(DIR + 'usecols.pkl', 'wb') as f:
        pickle.dump(usecols, f, -1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=871)

    # {'colsample_bytree': 0.9, 'learning_rate': 0.01, 'max_bin': 255, 'max_depth': -1, 'metric': 'binary_logloss', 'min_child_weight': 5, 'min_split_gain': 0.01, 'num_leaves': 63, 'objective': 'cross_entropy', 'reg_alpha': 1, 'seed': 114, 'subsample': 1.0, 'subsample_freq': 1, 'verbose': -1}
    all_params = {'min_child_weight': [3, 5, 10],
                  'subsample': [1.0, 0.7],
                  'subsample_freq': [1],
                  'seed': [114],
                  'colsample_bytree': [0.9, 0.7],
                  'learning_rate': [0.1],
                  'max_depth': [-1],
                  'min_split_gain': [0.01, 0],
                  'reg_alpha': [1, 0],
                  'max_bin': [255],
                  'num_leaves': [31, 63, 127],
                  'objective': ['binary'],
                  'metric': ['auc'],
                  'scale_pos_weight': [scale_pos_weight],
                  'verbose': [-1],
                  }

    use_score = 0
    min_score = (100, 100, 100)

    for params in tqdm(list(ParameterGrid(all_params))):
        cnt = -1
        list_score = []
        list_score2 = []
        list_best_iter = []
        all_pred = np.zeros(y_train.shape[0])
        for train, test in cv.split(x_train, y_train):
            cnt += 1
            trn_x = x_train.iloc[train]
            val_x = x_train.iloc[test]
            trn_y = y_train[train]
            val_y = y_train[test]

            train_data = lgb.Dataset(trn_x, label=trn_y)
            test_data = lgb.Dataset(val_x, label=val_y)
            clf = lgb.train(params,
                            train_data,
                            10000,  # params['n_estimators'],
                            early_stopping_rounds=30,
                            valid_sets=[test_data],
                            # feval=cst_metric_xgb,
                            # callbacks=[callback],
                            verbose_eval=10
                            )
            pred = clf.predict(val_x)

            all_pred[test] = pred

            _score2 = log_loss(val_y, pred)
            _score = - roc_auc_score(val_y, pred)

            logger.info('   _score: %s' % _score)
            logger.info('   _score2: %s' % _score2)

            list_score.append(_score)
            list_score2.append(_score2)

            if clf.best_iteration != 0:
                list_best_iter.append(clf.best_iteration)
            else:
                list_best_iter.append(params['n_estimators'])

            with open(DIR + 'train_cv_pred_%s.pkl' % cnt, 'wb') as f:
                pickle.dump(pred, f, -1)
            with open(DIR + 'model_%s.pkl' % cnt, 'wb') as f:
                pickle.dump(clf, f, -1)
            del trn_x
            del clf
            gc.collect()
            break
        with open(DIR + 'train_cv_tmp.pkl', 'wb') as f:
            pickle.dump(all_pred, f, -1)

        logger.info('trees: {}'.format(list_best_iter))
        #trees = np.mean(list_best_iter, dtype=int)
        score = (np.mean(list_score), np.min(list_score), np.max(list_score))
        score2 = (np.mean(list_score2), np.min(list_score2), np.max(list_score2))

        logger.info('param: %s' % (params))
        logger.info('cv: {})'.format(list_score))
        logger.info('cv2: {})'.format(list_score2))

        logger.info('loss: {} (avg min max {})'.format(score[use_score], score))
        logger.info('qwk: {} (avg min max {})'.format(score2[use_score], score2))

        if min_score[use_score] > score[use_score]:
            min_score = score
            min_params = params
        logger.info('best score: {} {}'.format(min_score[use_score], min_score))

    gc.collect()
    train_data = lgb.Dataset(x_train, label=y_train)
    logger.info('train start')
    clf = lgb.train(min_params,
                    train_data,
                    int(np.mean(list_best_iter) * 1.1),
                    valid_sets=[train_data],
                    verbose_eval=30
                    )
    logger.info('train end')
    with open(DIR + 'model.pkl', 'wb') as f:
        pickle.dump(clf, f, -1)
    del x_train
    gc.collect()

    logger.info('save end')


def predict():
    with open(DIR + 'model.pkl', 'rb') as f:
        clf = pickle.load(f)

    with open(DIR + 'usecols.pkl', 'rb') as f:
        usecols = pickle.load(f)

    imp = pd.DataFrame(clf.feature_importance(), columns=['imp'])
    imp['col'] = usecols
    n_features = imp.shape[0]
    imp = imp.sort_values('imp', ascending=False)
    imp.to_csv(DIR + 'feature_importances.csv')
    logger.info('imp use {} {}'.format(imp[imp.imp > 0].shape, n_features))

    df = load_test_data()
    logger.info('data size {}'.format(df.shape))

    for col in usecols:
        if col not in df.columns.values:
            df[col] = np.zeros(df.shape[0])
            logger.info('no col %s' % col)

    x_test = df[usecols].fillna(-100)
    if x_test.shape[1] != n_features:
        raise Exception('Not match feature num: %s %s' % (x_test.shape[1], n_features))

    logger.info('test load end')

    p_test = clf.predict(x_test)
    with open(DIR + 'test_tmp_pred.pkl', 'wb') as f:
        pickle.dump(p_test, f, -1)

    logger.info('test save end')

    sub = pd.DataFrame()

    sub['click_id'] = df['click_id']
    sub['is_attributed'] = p_test
    sub.to_csv(DIR + 'submit.csv', index=False)
    logger.info('exit')


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
    predict()
