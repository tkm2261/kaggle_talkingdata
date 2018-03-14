
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, concatenate, BatchNormalization, Lambda, Activation, GRU, SimpleRNN, RNN
from keras.layers import CuDNNGRU, CuDNNLSTM, StackedRNNCells, MaxPooling1D, AvgPool1D, LeakyReLU
from keras.layers.wrappers import TimeDistributed
from keras import backend as K
import tensorflow as tf

MAX_SEQUENCE_LENGTH = 5

LIST_COL = ['app', 'device', 'os', 'channel', 'hour',  # 'sum_attr', 'last_attr',
            #'ip',
            'avg_ip', 'avg_app', 'avg_device', 'avg_os', 'avg_channel', 'avg_day', 'avg_hour', 'avg_ipdayhour']

LIST_CONV_COL = ['avg_avg', 'max_avg', 'min_avg']

LIST_ALL_COL = LIST_COL + LIST_CONV_COL + ['is_attributed']

LIST_CAT_COL = ['app', 'device', 'os', 'channel', 'hour']

LIST_FLOAT_COL = ['avg_ip', 'avg_app', 'avg_device', 'avg_os', 'avg_channel', 'avg_day', 'avg_hour', 'avg_ipdayhour']

LIST_DATA_COL = LIST_CAT_COL + LIST_FLOAT_COL

MAP_COL_NUM = {'app': 706, 'device': 3475, 'os': 800, 'channel': 202, 'hour': 24}


def custom_objective(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)


def auc(y_true, y_pred):
    score, up_opt = tf.metrics.auc(y_true, y_pred)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score


def get_lstm_sin(first_dences=[64, 64, 16],
                 is_first_bn=False,
                 lstm_size=64,
                 lstm_dropout=0.15,
                 lstm_recurrent_dropout=0.15,
                 # gru_dropout=0.15,
                 # gru_recurrent_dropout=0.15,
                 # rnn_dropout=0.15,
                 # rnn_recurrent_dropout=0.15,
                 last_dences=[16, 8],
                 is_last_bn=False,
                 learning_rate=1.0e-3,
                 ):

    inputs = {col: Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32' if col in LIST_CAT_COL else 'float32',
                         name=f'{col}_input') for col in LIST_ALL_COL}

    one_hots = [Lambda(K.one_hot, arguments={'num_classes': MAP_COL_NUM[col] + 1}, output_shape=(MAX_SEQUENCE_LENGTH, MAP_COL_NUM[col] + 1))(inputs[col])
                for col in LIST_CAT_COL]
    avgs = concatenate([Lambda(K.expand_dims)(inputs[col]) for col in LIST_CONV_COL])

    others = concatenate([Lambda(K.expand_dims)(inputs[col])
                          for col in LIST_ALL_COL if col not in LIST_FLOAT_COL + LIST_CAT_COL], axis=2)

    floats = concatenate([Lambda(K.expand_dims)(inputs[col]) for col in LIST_FLOAT_COL], axis=2)
    old_target = Lambda(K.expand_dims)(inputs['is_attributed'])

    out = concatenate(one_hots + [others] + [floats] + [old_target], axis=2)

    for i, size in enumerate(first_dences):
        out = Dense(size, name=f'first_dence_{i}_{size}')(out)
        if is_first_bn:
            out = BatchNormalization(name=f'first_bn_{i}_{size}')(out)
        out = LeakyReLU()(out)  #
        #out = Activation('relu')(out)

    out = concatenate([out, avgs], axis=2)

    lstm = LSTM(lstm_size,
                dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout,
                name='lstm')(out)

    gru = CuDNNGRU(16, name='gru')(out)
    #rnn = SimpleRNN(lstm_size, name='rnn')(out)

    merged = concatenate([lstm, gru])

    for i, size in enumerate(last_dences):
        merged = Dense(size, name=f'last_de1_{i}_{size}')(merged)
        if is_last_bn:
            merged = BatchNormalization(name=f'last_bn1_{i}_{size}')(merged)
        #merged = LeakyReLU()(merged)  #
        merged = Activation('relu')(merged)

    preds = Dense(1, activation='sigmoid', name='last1')(merged)

    model = Model([inputs[col] for col in LIST_ALL_COL], preds)
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=learning_rate),
                  metrics=[auc])
    return model


if __name__ == '__main__':
    model = get_lstm_sin()
