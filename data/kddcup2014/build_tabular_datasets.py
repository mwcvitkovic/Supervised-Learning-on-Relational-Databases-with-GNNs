import os

import pandas as pd

from __init__ import data_root

db_name = 'kddcup2014'

if __name__ == '__main__':
    raw_data_dir = os.path.join(data_root, 'raw_data', db_name)

    projects = pd.read_csv(os.path.join(raw_data_dir, 'projects.csv'), index_col=False)
    essays = pd.read_csv(os.path.join(raw_data_dir, 'essays.csv'), index_col=False)
    outcomes = pd.read_csv(os.path.join(raw_data_dir, 'outcomes.csv'), usecols=['projectid', 'is_exciting'],
                           index_col=False)

    # Add target to projects and rename
    projects = projects.merge(outcomes, how='outer', on='projectid')
    projects.set_index('projectid', inplace=True)
    projects.rename(columns={'is_exciting': 'TARGET'}, inplace=True)
    mapping = {'t': 1.0, 'f': 0.0}
    projects['TARGET'] = projects['TARGET'].replace(mapping)

    # Fix latlong
    projects['school_latlong'] = projects[['school_latitude', 'school_longitude']].apply(tuple, axis=1)
    projects.drop(['school_latitude', 'school_longitude'], axis=1, inplace=True)

    # Drop redundant column
    essays.drop('teacher_acctid', axis=1, inplace=True)

    cols = ['TARGET'] + [c for c in projects.columns.to_list() if c != 'TARGET']
    projects = projects[cols]

    ds_name = 'kddcup2014_main_table'
    out_dir = os.path.join(data_root, 'tabular', ds_name)
    os.makedirs(out_dir, exist_ok=True)
    tabular_file_path = os.path.join(out_dir, ds_name + '.csv.gz')
    projects.to_csv(tabular_file_path)

    ds_name = 'kddcup2014_main_table_and_essays'
    p_and_e = projects.merge(essays, how='inner', on='projectid')
    p_and_e.set_index('projectid', inplace=True)
    out_dir = os.path.join(data_root, 'tabular', ds_name)
    os.makedirs(out_dir, exist_ok=True)
    tabular_file_path = os.path.join(out_dir, ds_name + '.csv.gz')
    p_and_e.to_csv(tabular_file_path)
