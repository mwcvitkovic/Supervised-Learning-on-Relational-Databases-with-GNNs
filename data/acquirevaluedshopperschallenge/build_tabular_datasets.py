import os

import numpy as np
import pandas as pd

from __init__ import data_root

db_name = 'acquirevaluedshopperschallenge'
ds_name = 'acquirevaluedshopperschallenge_main_table'

if __name__ == '__main__':
    raw_data_dir = os.path.join(data_root, 'raw_data', db_name)

    train_history = pd.read_csv(os.path.join(raw_data_dir, 'trainHistory.csv'), index_col='id')
    test_history = pd.read_csv(os.path.join(raw_data_dir, 'testHistory.csv'), index_col='id')

    train_history.drop('repeattrips', axis=1, inplace=True)
    train_history['TARGET'] = train_history.transform({'repeater': lambda x: True if x == 't' else False})
    train_history.drop('repeater', axis=1, inplace=True)
    test_history['TARGET'] = np.nan
    history = train_history.append(test_history)
    cols = ['TARGET'] + [c for c in history.columns.to_list() if c != 'TARGET']
    history = history[cols]

    out_dir = os.path.join(data_root, 'tabular', ds_name)
    os.makedirs(out_dir, exist_ok=True)
    tabular_file_path = os.path.join(out_dir, ds_name + '.csv.gz')
    history.to_csv(tabular_file_path)
