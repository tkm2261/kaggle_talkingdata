import pandas as pd
import numpy as np
import glob
from tqdm import tqdm


def load_train_data():
    df = pd.concat([pd.read_csv(path)
                    for path in tqdm(glob.glob('../data/dmt_train/*.csv.gz'))], ignore_index=True, copy=False)


def main():
    load_train_data()


if __name__ == '__main__':
    main()
