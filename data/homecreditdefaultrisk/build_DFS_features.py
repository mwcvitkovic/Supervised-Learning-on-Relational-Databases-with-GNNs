"""
Adapted from https://www.kaggle.com/willkoehrsen/home-credit-default-risk-feature-tools and
https://www.kaggle.com/willkoehrsen/automated-feature-engineering-basics.
"""

import os

import featuretools as ft
import numpy as np
import pandas as pd

from __init__ import data_root
from data.utils import get_db_info, set_entity_variable_types, compute_and_save_dfs_features

db_name = 'homecreditdefaultrisk'
dp_limit = None
max_depth = 2
n_jobs = 1
chunk_size = 1000

if __name__ == '__main__':
    print('Loading data')
    data_dir = os.path.join(data_root, 'raw_data', db_name)
    db_info = get_db_info(db_name)

    app_train = pd.read_csv(os.path.join(data_dir, 'application_train.csv'), nrows=dp_limit, index_col=False)
    app_test = pd.read_csv(os.path.join(data_dir, 'application_test.csv'), nrows=dp_limit, index_col=False)
    bureau = pd.read_csv(os.path.join(data_dir, 'bureau.csv'), nrows=dp_limit, index_col=False)
    bureau_balance = pd.read_csv(os.path.join(data_dir, 'bureau_balance.csv'), nrows=dp_limit, index_col=False)
    cash = pd.read_csv(os.path.join(data_dir, 'POS_CASH_balance.csv'), nrows=dp_limit, index_col=False)
    credit = pd.read_csv(os.path.join(data_dir, 'credit_card_balance.csv'), nrows=dp_limit, index_col=False)
    previous = pd.read_csv(os.path.join(data_dir, 'previous_application.csv'), nrows=dp_limit, index_col=False)
    installments = pd.read_csv(os.path.join(data_dir, 'installments_payments.csv'), nrows=dp_limit, index_col=False)

    app_train['TARGET'] = app_train['TARGET'].astype(np.float64)
    app_test["TARGET"] = np.nan
    app = app_train.append(app_test, ignore_index=True)

    # Make entity set
    es = ft.EntitySet(id='applications')

    # Entities with a unique index
    es = es.entity_from_dataframe(entity_id='app',
                                  dataframe=app,
                                  index='SK_ID_CURR')
    es = es.entity_from_dataframe(entity_id='bureau',
                                  dataframe=bureau,
                                  index='SK_ID_BUREAU')
    es = es.entity_from_dataframe(entity_id='previous',
                                  dataframe=previous,
                                  index='SK_ID_PREV')
    # Entities that do not have a unique index
    es = es.entity_from_dataframe(entity_id='bureau_balance',
                                  dataframe=bureau_balance,
                                  make_index=True,
                                  index='bureaubalance_index')
    es = es.entity_from_dataframe(entity_id='cash',
                                  dataframe=cash,
                                  make_index=True,
                                  index='cash_index')
    es = es.entity_from_dataframe(entity_id='installments',
                                  dataframe=installments,
                                  make_index=True,
                                  index='installments_index')
    es = es.entity_from_dataframe(entity_id='credit',
                                  dataframe=credit,
                                  make_index=True,
                                  index='credit_index')

    # Define relationships (except featuretools can't handle multiple paths from a parent to a child)
    r_bureau_app = ft.Relationship(es['app']['SK_ID_CURR'], es['bureau']['SK_ID_CURR'])
    r_bureaubalance_bureau = ft.Relationship(es['bureau']['SK_ID_BUREAU'], es['bureau_balance']['SK_ID_BUREAU'])
    r_previousapp_app = ft.Relationship(es['app']['SK_ID_CURR'], es['previous']['SK_ID_CURR'])
    # r_cashbalance_app = ft.Relationship(es['app']['SK_ID_CURR'], es['cash']['SK_ID_PREV'])
    r_cashbalance_previousapp = ft.Relationship(es['previous']['SK_ID_PREV'], es['cash']['SK_ID_PREV'])
    # r_credit_app = ft.Relationship(es['app']['SK_ID_CURR'], es['credit']['SK_ID_CURR'])
    r_credit_previousapp = ft.Relationship(es['previous']['SK_ID_PREV'], es['credit']['SK_ID_PREV'])
    # r_installments_app = ft.Relationship(es['app']['SK_ID_CURR'], es['installments']['SK_ID_CURR'])
    r_installments_previousapp = ft.Relationship(es['previous']['SK_ID_PREV'], es['installments']['SK_ID_PREV'])

    # Add in the defined relationships
    es = es.add_relationships([r_bureau_app,
                               r_bureaubalance_bureau,
                               r_previousapp_app,
                               # r_cashbalance_app,
                               r_cashbalance_previousapp,
                               # r_credit_app,
                               r_credit_previousapp,
                               # r_installments_app,
                               r_installments_previousapp])

    # Fix up variable types
    set_entity_variable_types(es['app'], 'Application', db_info)
    set_entity_variable_types(es['bureau'], 'Bureau', db_info)
    set_entity_variable_types(es['previous'], 'PreviousApplication', db_info)
    set_entity_variable_types(es['bureau_balance'], 'BureauBalance', db_info)
    set_entity_variable_types(es['cash'], 'CashBalance', db_info)
    set_entity_variable_types(es['installments'], 'InstallmentPayment', db_info)
    set_entity_variable_types(es['credit'], 'CreditBalance', db_info)

    compute_and_save_dfs_features(es, 'app', db_name, max_depth, n_jobs, chunk_size)
