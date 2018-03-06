
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, concatenate, BatchNormalization, Lambda
from keras.layers.wrappers import TimeDistributed
from keras import backend as K
import tensorflow as tf

MAX_SEQUENCE_LENGTH = 100

LIST_DATA_COL = ['list_app', 'list_device', 'list_os', 'list_ch', 'list_timediff',
                 'list_month', 'list_day', 'list_dayofweek', 'list_hour', 'list_sum_attr', 'list_attr']

LIST_CAT_COL = ['list_app', 'list_device', 'list_os', 'list_ch',
                'list_month', 'list_day', 'list_dayofweek', 'list_hour']

LIST_FLOAT_COL = ['list_timediff', 'list_sum_attr', 'list_attr']

# MAP_COL_NUM = {'list_app': 768, 'list_device': 4227, 'list_os': 956, 'list_ch': 500}
MAP_COL_NUM = {'list_app': 706, 'list_device': 3475, 'list_os': 800, 'list_ch': 202,
               'list_month': 12, 'list_day': 31, 'list_dayofweek': 7, 'list_hour': 24}

LIST_COL_NUM = [768, 4227, 956, 500]


def custom_objective(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)


def auc(y_true, y_pred):
    score, up_opt = tf.metrics.auc(y_true, y_pred)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score


def get_lstm():
    np.random.seed(0)

    inputs = {col: Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32' if col in LIST_FLOAT_COL else 'int32',
                         name=f'{col}_input') for col in LIST_DATA_COL}

    one_hots = [Lambda(K.one_hot, arguments={'num_classes': MAP_COL_NUM[col] + 1}, output_shape=(MAX_SEQUENCE_LENGTH, MAP_COL_NUM[col] + 1))(inputs[col])
                for col in LIST_CAT_COL]

    floats = [Lambda(K.expand_dims)(inputs[col]) for col in LIST_FLOAT_COL]
    all_input = concatenate(one_hots + floats, axis=2)

    out = Dense(192, activation='relu')(all_input)
    out = Dense(64, activation='relu')(out)
    out = Dense(64, activation='relu')(out)

    lstm_layer = LSTM(64, dropout=0.15, recurrent_dropout=0.15, return_sequences=True,
                      name='lstm', input_shape=(MAX_SEQUENCE_LENGTH, None))(out)

    merged = TimeDistributed(BatchNormalization(name='BN1'), name='TD-BN1')(lstm_layer)
    merged = TimeDistributed(Dense(32, activation='relu', name='D1'), name='TD-D1')(merged)
    merged = TimeDistributed(BatchNormalization(name='BN2'), name='TD-BN2')(merged)
    preds = TimeDistributed(Dense(1, activation='sigmoid', name='last'), name='TD-LST')(merged)

    model = Model([inputs[col] for col in LIST_DATA_COL], preds)
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=1e-3),
                  metrics=[auc])
    return model


if __name__ == '__main__':
    model = get_lstm()
