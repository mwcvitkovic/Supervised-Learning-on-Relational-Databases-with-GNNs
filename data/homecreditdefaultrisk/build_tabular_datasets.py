import os

import numpy as np
import pandas as pd

from __init__ import data_root

db_name = 'homecreditdefaultrisk'
ds_name = 'homecreditdefaultrisk_main_table'

if __name__ == '__main__':
    raw_data_dir = os.path.join(data_root, 'raw_data', db_name)

    app_train = pd.read_csv(os.path.join(raw_data_dir, 'application_train.csv'), index_col='SK_ID_CURR')
    app_test = pd.read_csv(os.path.join(raw_data_dir, 'application_test.csv'), index_col='SK_ID_CURR')

    app_train['TARGET'] = app_train['TARGET'].astype(np.float64)
    app_test['TARGET'] = np.nan
    app = app_train.append(app_test)
    cols = ['TARGET'] + [c for c in app.columns.to_list() if c != 'TARGET']
    app = app[cols]

    out_dir = os.path.join(data_root, 'tabular', ds_name)
    os.makedirs(out_dir, exist_ok=True)
    tabular_file_path = os.path.join(out_dir, ds_name + '.csv.gz')
    app.to_csv(tabular_file_path)
