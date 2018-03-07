import pandas as pd
import numpy as np
import glob
import json
from tqdm import tqdm
from lstm import get_lstm, MAX_SEQUENCE_LENGTH, LIST_DATA_COL, LIST_CONV_COL


def read_csv(path):
    df = pd.read_csv(path)
    for col in LIST_DATA_COL + ['list_target']:
        if col in LIST_CONV_COL:
            c = col.replace('avg_', '')
            postfix = col.split('_')[-1]
            map_d = pd.read_csv(f'../data/mst_{postfix}.csv', index_col=postfix).to_dict()[f'avg_{postfix}']
            df[col] = df[c].apply(lambda x: [map_d.get(i, -1) for i in x])
        else:
            df[col].fillna('[]', inplace=True)
            df[col] = df[col].apply(json.loads)

    print(path)
    return df


def load_train_data():
    df = pd.concat(list(map(read_csv, glob.glob('../data/dmt_train/*.csv.gz')[:1])),
                   ignore_index=True, copy=False)
    import pdb
    pdb.set_trace()


def main():
    load_train_data()


if __name__ == '__main__':
    main()
