
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, concatenate, BatchNormalization, Lambda, Activation, GRU, SimpleRNN, RNN
from keras.layers import CuDNNGRU, CuDNNLSTM, StackedRNNCells, MaxPooling1D, AvgPool1D
from keras.layers.wrappers import TimeDistributed
from keras import backend as K
import tensorflow as tf


LIST_FLOAT_COL = [  # 'hour',
    #'avg_ip',
    #'sum_attr', 'last_attr',
    'avg_app',
    'avg_device', 'avg_os', 'avg_channel', 'avg_day', 'avg_hour']

LIST_CAT_COL = ['app', 'device', 'os', 'channel', 'hour']

LIST_DATA_COL = LIST_FLOAT_COL + LIST_CAT_COL

LIST_CONV_COL = None

MAP_COL_NUM = {'app': 706, 'device': 3475, 'os': 800, 'channel': 202, 'hour': 24}


def custom_objective(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)


def auc(y_true, y_pred):
    score, up_opt = tf.metrics.auc(y_true, y_pred)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score


def get_dense(first_dences=[64, 32, 32, 8],
              learning_rate=1.0e-3,
              ):

    floats = input_float = Input(shape=(len(LIST_FLOAT_COL),), dtype='float32', name='input')
    inputs = {col: Input(shape=(1, ), dtype='int32',
                         name=f'{col}_input') for col in LIST_CAT_COL}

    one_hots = [Lambda(tf.one_hot, arguments={'depth': MAP_COL_NUM[col] + 1, 'axis': -1}, output_shape=(1, MAP_COL_NUM[col] + 1))(inputs[col])
                for col in LIST_CAT_COL]
    #one_hots = [Lambda(lambda x: x[0, :])(ele) for ele in one_hots]
    one_hots = concatenate(one_hots)
    _floats = Lambda(K.expand_dims, arguments={'axis': 1})(floats)
    out = concatenate([one_hots, _floats])
    out = Lambda(lambda x: x[:, 0, :])(out)

    _floats = Lambda(K.expand_dims)(floats)
    max_avg = MaxPooling1D(pool_size=len(LIST_FLOAT_COL))(_floats)
    max_avg = Lambda(lambda x: x[:, :, 0])(max_avg)

    avg_avg = AvgPool1D(pool_size=len(LIST_FLOAT_COL))(_floats)
    avg_avg = Lambda(lambda x: x[:, :, 0])(avg_avg)

    for i, size in enumerate(first_dences):
        out = Dense(size, name=f'first_dence_{i}_{size}')(out)
        out = Activation('relu')(out)

    out = concatenate([max_avg, avg_avg, out])

    preds = Dense(1, activation='sigmoid', name='last1')(out)

    model = Model([input_float] + [inputs[col] for col in LIST_CAT_COL], preds)
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=learning_rate),
                  metrics=[auc])
    return model


if __name__ == '__main__':
    model = get_dense()
