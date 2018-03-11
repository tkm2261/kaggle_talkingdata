import json
from sklearn.model_selection import ParameterGrid


if __name__ == '__main__':

    all_params = {'first_dences': [[128, 16], [64, 64, 32], [128, 64, 64, 32, 32, 16], [16, 16, 16, 8]],
                  #'is_first_bn': [False, True],
                  #'lstm_size': [16, 32],
                  #'lstm_dropout': [0.15, 0.3],
                  #'lstm_recurrent_dropout': [0.15, 0.3],
                  #'gru_dropout': [0.15],
                  #'gru_recurrent_dropout': [0.15],
                  #'rnn_dropout': [0.15],
                  #'rnn_recurrent_dropout': [0.15],
                  #'last_dences': [[16, 8], [64, 32]],
                  #'is_last_bn': [False, True],
                  'learning_rate': [1.0e-3]}
    ff = open('run.sh', 'w')
    for i, params in enumerate(ParameterGrid(all_params)):
        with open(f'params/param_{i}.json', 'w') as f:
            f.write(json.dumps(params, sort_keys=True, indent=2) + '\n')
        ff.write(f'python train_dense.py params/param_{i}.json\n')
