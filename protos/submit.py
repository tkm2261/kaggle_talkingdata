import glob
import pandas as pd

df = pd.concat(map(pd.read_csv, glob.glob('submit/*csv')), axis=0, ignore_index=True)

df.sort_values('click_id', inplace=True)

"""
a = df.groupby('click_id')['click_id'].count().sort_values(ascending=False)
print(df.shape, a.shape)
print(a.head())
df = df.set_index('click_id')

sub = pd.read_csv('../input/sample_submission.csv').set_index('click_id')
import pdb
pdb.set_trace()

df = df.loc[sub.index.values, :].reset_index()
"""

df.to_csv('submit.csv', index=False)
