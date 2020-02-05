import os

import featuretools as ft
import numpy as np
import pandas as pd
from featuretools.variable_types import Categorical, Numeric, Datetime
from joblib import Parallel, delayed

from __init__ import data_root
from data.utils import get_db_info, run_dfs

db_name = 'acquirevaluedshopperschallenge'
dp_limit = None
part_size = 5000
chunk_size = 1000
max_depth = 2
n_jobs = 20


def make_es_and_dfs_for_partition(history_part, transactions_part):
    # Make entity set
    es = ft.EntitySet(id='history')

    es = es.entity_from_dataframe(entity_id='history',
                                  dataframe=history_part,
                                  index='id',
                                  variable_types={'chain': Categorical,
                                                  'market': Categorical,
                                                  'offer': Categorical,
                                                  'offerdate': Datetime,
                                                  'TARGET': Categorical,
                                                  'category': Categorical,
                                                  'quantity': Numeric,
                                                  'company': Categorical,
                                                  'offervalue': Numeric,
                                                  'brand': Categorical})
    es = es.entity_from_dataframe(entity_id='transactions',
                                  dataframe=transactions_part,
                                  index='id',
                                  make_index=True,
                                  variable_types={'chain': Categorical,
                                                  'dept': Categorical,
                                                  'category': Categorical,
                                                  'company': Categorical,
                                                  'brand': Categorical,
                                                  'date': Datetime,
                                                  'productsize': Numeric,
                                                  'productmeasure': Categorical,
                                                  'purchasequantity': Numeric,
                                                  'purchaseamount': Numeric}
                                  )

    # Define relationship
    r_transaction_history = ft.Relationship(es['history']['id'], es['transactions']['historyid'])

    # Add in the defined relationships
    es = es.add_relationships([r_transaction_history])

    print('Saving only main table features (no DFS)')
    agg_primitives = []
    trans_primitives = []
    ignore_variables = {}
    main_table_features = run_dfs(es, 'history', agg_primitives, trans_primitives, ignore_variables, max_depth, 1,
                                  chunk_size)

    print('Running DFS with default primitives')
    agg_primitives = ["sum", "std", "max", "skew", "min", "mean", "count", "percent_true", "num_unique", "mode"]
    trans_primitives = ["day", "year", "month", "weekday", "haversine", "num_words", "num_characters"]
    ignore_variables = {}
    smaller_dfs_features = run_dfs(es, 'history', agg_primitives, trans_primitives, ignore_variables, max_depth, 1,
                                   chunk_size)

    print('Running DFS with all primitives')
    # Use every possible primitive
    agg_primitives = ['median', 'mode', 'time_since_last',  # deleted n_most_common to avoid featuretools bug
                      'all', 'mean', 'percent_true', 'any',
                      'sum', 'num_true', 'time_since_first', 'last', 'count', 'num_unique', 'trend', 'std', 'max',
                      'skew', 'min', 'avg_time_between']
    trans_primitives = None
    ignore_variables = {}
    larger_dfs_features = run_dfs(es, 'history', agg_primitives, trans_primitives, ignore_variables, max_depth, 1,
                                  chunk_size)

    return main_table_features, smaller_dfs_features, larger_dfs_features


if __name__ == '__main__':
    print('Loading data')
    data_dir = os.path.join(data_root, 'raw_data', db_name)
    db_info = get_db_info(db_name)

    train_history = pd.read_csv(os.path.join(data_dir, 'trainHistory.csv'), nrows=dp_limit, index_col=False)
    test_history = pd.read_csv(os.path.join(data_dir, 'testHistory.csv'), nrows=dp_limit, index_col=False)
    transactions = pd.read_csv(os.path.join(data_dir, 'transactions.csv'), nrows=dp_limit, index_col=False)
    offers = pd.read_csv(os.path.join(data_dir, 'offers.csv'), nrows=dp_limit, index_col=False)

    test_history['repeater'] = np.nan
    train_history.drop('repeattrips', axis=1, inplace=True)
    history = train_history.append(test_history, ignore_index=True)
    history.rename(columns={'repeater': 'TARGET'}, inplace=True)
    history = history.merge(offers, on='offer', how='inner')
    mapping = {'t': 1.0, 'f': 0.0}
    history['TARGET'] = history['TARGET'].replace(mapping)
    transactions.rename(columns={'id': 'historyid'}, inplace=True)


    def history_transaction_generator(part_size):
        for i in range(0, history.shape[0], part_size):
            history_part = history[i:i + part_size]
            transaction_part = transactions.loc[transactions['historyid'].isin(history_part['id'])]
            # assert transaction_part.shape[0] > 0
            yield (history_part, transaction_part)


    part_features = Parallel(n_jobs=n_jobs,
                             verbose=1)(
        delayed(make_es_and_dfs_for_partition)(hp, tp) for hp, tp in history_transaction_generator(part_size))

    print('Concatenating and saving features')
    write_data_dir = os.path.join(data_root, db_name)
    main_table_features, smaller_dfs_features, larger_dfs_features = list(zip(*part_features))

    main_table_features = pd.concat(main_table_features)
    assert history.shape[0] == main_table_features.shape[0]
    assert main_table_features.index.is_unique
    dfs_feat_file = os.path.join(write_data_dir, f'tabular_features_main_table_{db_name}.pkl')
    main_table_features.to_pickle(dfs_feat_file)

    smaller_dfs_features = pd.concat(smaller_dfs_features)
    assert history.shape[0] == smaller_dfs_features.shape[0]
    assert smaller_dfs_features.index.is_unique
    dfs_feat_file = os.path.join(write_data_dir, f'tabular_features_dfs_smaller_{db_name}.pkl')
    smaller_dfs_features.to_pickle(dfs_feat_file)

    larger_dfs_features = pd.concat(larger_dfs_features)
    assert history.shape[0] == larger_dfs_features.shape[0]
    assert larger_dfs_features.index.is_unique
    dfs_feat_file = os.path.join(write_data_dir, f'tabular_features_dfs_larger_{db_name}.pkl')
    larger_dfs_features.to_pickle(dfs_feat_file)
