import json
import os
import subprocess
import sys
import time

import docker
import featuretools as ft
import networkx as nx
import numpy as np
import pandas as pd
from featuretools.variable_types import Index, Id, Categorical, Numeric, Text, Datetime, Ordinal
from neo4j import GraphDatabase
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm

from __init__ import project_root, data_root
from data.data_encoders import ScalarRobustScalerEnc, ScalarPowerTransformerEnc, ScalarQuantileTransformerEnc, \
    ScalarQuantileOrdinalEnc, TextSummaryScalarEnc, TfidfEnc


def populate_db_info(db_name, db_info):
    driver = get_neo4j_db_driver(db_name)
    with driver.session() as session:
        for node_type, features in db_info['node_types_and_features'].items():
            for feature_name, feature_info in features.items():
                if feature_info['type'] == 'CATEGORICAL':
                    query = 'MATCH (n:{0}) RETURN DISTINCT n.{1} ORDER BY n.{1} ;'.format(node_type, feature_name)
                    print('running: {}'.format(query))
                    distinct_values = session.run(query).value()

                    nulls = distinct_values.count(None)
                    if nulls == 1:
                        distinct_values.remove(None)
                    elif nulls > 1:
                        raise ValueError

                    feature_info['sorted_values'] = distinct_values
                    feature_info['n_distinct_values'] = len(distinct_values)

                elif feature_info['type'] == 'SCALAR':
                    query = 'MATCH (n:{0}) RETURN n.{1} ;'.format(node_type, feature_name)
                    print('running: {}'.format(query))
                    scalars = session.run(query).value()

                    scalars = pd.Series([s for s in scalars if s is not None])

                    enc = ScalarRobustScalerEnc()
                    enc.fit(scalars)
                    feature_info['RobustScaler_center_'] = float(enc.scaler.center_)
                    feature_info['RobustScaler_scale_'] = float(enc.scaler.scale_)

                    enc = ScalarPowerTransformerEnc()
                    enc.fit(scalars)
                    feature_info['PowerTransformer_lambdas_'] = float(enc.scaler.lambdas_)
                    feature_info['PowerTransformer_scale_'] = float(enc.scaler._scaler.scale_)
                    feature_info['PowerTransformer_mean_'] = float(enc.scaler._scaler.mean_)
                    feature_info['PowerTransformer_var_'] = float(enc.scaler._scaler.var_)
                    feature_info['PowerTransformer_n_samples_seen_'] = int(enc.scaler._scaler.n_samples_seen_)

                    enc = ScalarQuantileTransformerEnc()
                    enc.fit(scalars)
                    feature_info['QuantileTransformer_n_quantiles_'] = int(enc.scaler.n_quantiles_)
                    feature_info['QuantileTransformer_quantiles_'] = enc.scaler.quantiles_[:, 0].tolist()
                    feature_info['QuantileTransformer_references_'] = enc.scaler.references_.tolist()

                    enc = ScalarQuantileOrdinalEnc()
                    enc.fit(scalars)
                    feature_info['KBinsDiscretizer_n_bins_'] = int(enc.disc.n_bins_)
                    feature_info['KBinsDiscretizer_bin_edges_'] = enc.disc.bin_edges_[0].tolist()

                elif feature_info['type'] == 'TEXT':
                    def rec_val_generator(generator):
                        while True:
                            try:
                                n = generator.__next__().value()
                                yield n if n else ''
                            except StopIteration:
                                return

                    query = 'MATCH (n:{0}) RETURN n.{1} ;'.format(node_type, feature_name)
                    print('running: {}'.format(query))
                    text_strings = session.run(query).records()
                    text_strings = pd.Series(rec_val_generator(text_strings))

                    enc = TextSummaryScalarEnc()
                    enc.fit(text_strings)
                    feature_info['RobustScaler_center_'] = enc.scaler.center_.tolist()
                    feature_info['RobustScaler_scale_'] = enc.scaler.scale_.tolist()

                    tfidf = TfidfEnc().get_new_base_enc()
                    tfidf.fit(text_strings)

                    vocabulary_ = {k: int(v) for k, v in tfidf.vocabulary_.items()}
                    feature_info['Tfidf_vocabulary_'] = vocabulary_
                    feature_info['Tfidf_idf_'] = tfidf.idf_.tolist()

                if '+++' in feature_name:
                    feature_name = feature_name.split('+++')[0]
                query = 'MATCH (n:{0}) WHERE n.{1} IS null RETURN count(n);'.format(node_type, feature_name)
                print('running: {}'.format(query))
                n_null_values = session.run(query).value()[0]
                feature_info['n_null_values'] = n_null_values

    return db_info


def build_db_info(db_name, db_info, test_dp_query, train_dp_query):
    db_info_path = os.path.join(project_root, 'data', db_name, '{}.db_info.json'.format(db_name))
    _ = get_db_container(db_name)
    populate_db_info(db_name, db_info)
    driver = get_neo4j_db_driver(db_name)
    with driver.session() as session:
        test_dp_ids = session.run(test_dp_query).value()
        train_dp_ids = session.run(train_dp_query).value()
    db_info['test_dp_ids'] = test_dp_ids
    db_info['train_dp_ids'] = train_dp_ids

    with open(db_info_path, 'w') as f:
        json.dump(db_info, f, indent=1, allow_nan=False)


def get_db_info(db_name):
    db_info_path = os.path.join(project_root, 'data', db_name, '{}.db_info.json'.format(db_name))
    with open(db_info_path, 'rb') as f:
        db_info = json.load(f)
    return db_info


def get_ds_info(ds_name):
    """
    ds_name can be the name of the dataset in data.tabular_ds_info.json, or a path to a .ds_info.json file
    """
    if ds_name == 'all_tabular_datasets':
        return {'processed': {'task': 'binary classification'}}
    elif os.path.isfile(ds_name):
        with open(ds_name, 'rb') as f:
            return json.load(f)
    else:
        ds_info_path = os.path.join(project_root, 'data', 'tabular_ds_info.json')
        with open(ds_info_path, 'rb') as f:
            ds_info = json.load(f)
        ds_info = ds_info[ds_name]
        meta_path = os.path.join(project_root, ds_info['processed']['ds_info'])
        with open(meta_path, 'rb') as f:
            meta = json.load(f)
        ds_info['meta'] = meta
        return ds_info


def create_datapoints_with_xargs(db_name, datapoint_ids, base_query, target_dir, n_jobs):
    """
    Hands the job of running create_datapoint_from_database in parallel over to the linux scheduler and xargs, for speed
        and memory improvements

    WARNING: if the tqdm process bar isn't increasing, there's probably a silent error in the called process.
        Copy and run the printed command in the terminal to debug.
    """
    temp_file = os.path.join(target_dir, 'jobs.txt')
    with open(temp_file, 'w') as f:
        f.writelines('\n'.join([str(dp_id) for dp_id in datapoint_ids]))
    cmd = "cat {} | xargs -I DPID --max-procs={} -n 1 {} -m {} {} DPID '{}' {}".format(temp_file,
                                                                                       n_jobs,
                                                                                       sys.executable,
                                                                                       'data.create_datapoint_from_database',
                                                                                       db_name,
                                                                                       base_query,
                                                                                       target_dir)
    print(cmd)
    with tqdm(total=len(datapoint_ids)) as pbar:
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                env=os.environ.copy())
        while proc.poll() is None:
            stream_out = str(proc.stdout.readline())
            # tqdm.write(stream_out)  # For debugging
            pbar.update()
        if proc.poll() != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
    os.remove(temp_file)


def get_db_container(db_name):
    """
    Grab the container id for this db's docker container, starting container if needed.
    If you can't connect, make sure the store_lock file is removed from the databases/ directory
    """
    ports_offsets_for_dbs = {
        'kddcup2014': 0,
        'avitocontextadclicks': 1000,
        'acquirevaluedshopperschallenge': 2000,
        'homecreditdefaultrisk': 3000,
    }
    source_dir = os.path.join(data_root, 'raw_data', db_name)
    plugin_dir = os.path.join(project_root,
                              'neo4j-apoc-plugins')  # These need to be downloaded, see https://github.com/neo4j-contrib/neo4j-apoc-procedures#using-apoc-with-the-neo4j-docker-image

    client = docker.from_env()
    # Return running container if exists...
    running_containers = client.containers.list()
    for rc in running_containers:
        if 'db_name' in rc.labels and rc.labels['db_name'] == db_name:
            return rc.id
    # ...otherwise start a new container
    container = client.containers.run('rdb-neo4j',
                                      detach=True,
                                      environment=["NEO4J_dbms_active__database={}.graph.db".format(db_name)],
                                      ports={'7473/tcp': 7473 + ports_offsets_for_dbs[db_name],
                                             '7474/tcp': 7474 + ports_offsets_for_dbs[db_name],
                                             '7687/tcp': 7687 + ports_offsets_for_dbs[db_name]},
                                      mounts=[docker.types.Mount(target='/data',
                                                                 source=source_dir,
                                                                 type='bind'),
                                              docker.types.Mount(target='/plugins',
                                                                 source=plugin_dir,
                                                                 type='bind')
                                              ],
                                      labels={'db_name': db_name})
    time.sleep(10)  # Let the DB initialize
    return container.id


def plot_graph(g: nx.Graph):
    import matplotlib.pyplot as plt
    nx.draw(g, with_labels=True)
    plt.savefig('g.png')
    plt.close()


def write_kaggle_submission_file(db_name, predictions, path):
    """
    predictions should be pandas dataframe with columns ['dp_id', 'prob']
    """
    if 'acquirevaluedshopperschallenge' in db_name:
        # Can upload this submission with:
        #    kaggle competitions submit -c acquire-valued-shoppers-challenge -f <path> -m "<message>"
        fieldnames = ['id', 'repeatProbability']
    elif 'homecreditdefaultrisk' in db_name:
        # Can upload this submission with:
        #    kaggle competitions submit -c home-credit-default-risk -f <path> -m "<message>"
        fieldnames = ['SK_ID_CURR', 'TARGET']
    elif 'kddcup2014' in db_name:
        # Can upload this submission with:
        #    kaggle competitions submit -c kdd-cup-2014-predicting-excitement-at-donors-choose -f <path> -m "<message>"
        fieldnames = ['projectid', 'is_exciting']
    else:
        raise ValueError(f'db_name {db_name} not recognized')

    predictions.columns = fieldnames
    predictions.to_csv(path, index=False)


def get_neo4j_db_driver(db_name):
    ports_for_dbs = {
        'kddcup2014': 7687,
        'acquirevaluedshopperschallenge': 9687,
        'homecreditdefaultrisk': 10687,
    }
    try:
        driver = GraphDatabase.driver("bolt://{}:{}".format('localhost', ports_for_dbs[db_name]))
        return driver
    except Exception as e:
        print('Is the neo4j database docker container running?  Run data.utils.get_db_container if not.')
        raise e


def train_val_split(dp_ids):
    return train_test_split(np.array(dp_ids), test_size=0.15, random_state=14)


def five_fold_split_iter(train_dp_ids):
    dp_ids = np.array(sorted(train_dp_ids))
    kf = KFold(n_splits=5,
               random_state=14,
               shuffle=True)
    for train_idx, val_idx in kf.split(dp_ids):
        yield dp_ids[train_idx], dp_ids[val_idx]


def set_entity_variable_types(entity, node_type, db_info):
    """
    Sets a featuretools Entity's variable types to match the variable types in db_info[node_type]
    """
    for var_name, var_type in entity.variable_types.items():
        if not var_type in [Index, Id]:
            feat_info = db_info['node_types_and_features'][node_type].get(var_name)
            if not feat_info:
                print(f'make sure {node_type}.{var_name} is set in variable_types, or is an Id')
            else:
                right_type = feat_info['type']
                new_type = None
                if right_type == 'CATEGORICAL':
                    new_type = Categorical
                elif right_type == 'SCALAR':
                    new_type = Numeric
                elif right_type == 'DATETIME':
                    assert var_type == Datetime
                elif right_type == 'TEXT':
                    assert var_type == Text
                else:
                    raise ValueError

                if new_type:
                    entity.convert_variable_type(var_name, new_type)


def set_dataframe_column_types(dfs_features, features):
    """
    Fixes the column types of the dataframe output by featuretools to match those of features
    (why this isn't done automatically I have no idea...)
    """
    assert len(dfs_features.columns) == len(features)
    for i in range(len(features)):
        col = dfs_features[dfs_features.columns[i]]
        feature = features[i]
        assert col.name == feature._name
        if feature.variable_type in [Categorical]:
            dfs_features[col.name] = col.astype('object')
        elif feature.variable_type in [Numeric, Ordinal]:
            dfs_features[col.name] = col.astype('float64')
        else:
            raise TypeError


def run_dfs(es, target_entity, agg_primitives, trans_primitives, ignore_variables, max_depth, n_jobs, chunk_size):
    dfs_object = ft.DeepFeatureSynthesis(target_entity_id=target_entity,
                                         entityset=es,
                                         agg_primitives=agg_primitives,
                                         trans_primitives=trans_primitives,
                                         ignore_variables=ignore_variables,
                                         max_depth=max_depth)
    features = dfs_object.build_features(verbose=True)
    # Remove target leakage
    features = [f for f in features if not ('TARGET' in f._name and 'TARGET' != f._name)]
    assert ['TARGET' in f._name for f in features].count(True) == 1
    dfs_features = ft.calculate_feature_matrix(features=features,
                                               entityset=es,
                                               chunk_size=chunk_size,
                                               n_jobs=n_jobs,
                                               verbose=True)
    # Fix the columns in default_dfs_features so they have the right pandas dtypes
    set_dataframe_column_types(dfs_features, features)

    print('Saving features')
    assert es[target_entity].shape[0] == dfs_features.shape[0]
    assert dfs_features.index.is_unique

    return dfs_features


def compute_and_save_dfs_features(es, target_entity, db_name, max_depth, n_jobs, chunk_size):
    data_dir = os.path.join(data_root, db_name)

    print('Saving only main table features (no DFS)')
    agg_primitives = []
    trans_primitives = []
    ignore_variables = {}
    dfs_features = run_dfs(es, target_entity, agg_primitives, trans_primitives, ignore_variables, max_depth, n_jobs,
                           chunk_size)
    dfs_feat_file = os.path.join(data_dir, f'tabular_features_main_table_{db_name}.pkl')
    dfs_features.to_pickle(dfs_feat_file)

    print('Running DFS with default primitives')
    agg_primitives = ["sum", "std", "max", "skew", "min", "mean", "count", "percent_true", "num_unique", "mode"]
    trans_primitives = ["day", "year", "month", "weekday", "haversine", "num_words", "num_characters"]
    ignore_variables = {}
    dfs_features = run_dfs(es, target_entity, agg_primitives, trans_primitives, ignore_variables, max_depth, n_jobs,
                           chunk_size)
    dfs_feat_file = os.path.join(data_dir, f'tabular_features_dfs_smaller_{db_name}.pkl')
    dfs_features.to_pickle(dfs_feat_file)

    print('Running DFS with all primitives')
    # Use every possible primitive
    agg_primitives = ['median', 'mode', 'time_since_last', 'n_most_common', 'all', 'mean', 'percent_true', 'any',
                      'sum', 'num_true', 'time_since_first', 'last', 'count', 'num_unique', 'trend', 'std', 'max',
                      'skew', 'min', 'avg_time_between']
    trans_primitives = ['cum_mean', 'diff', 'is_weekend', 'num_words', 'subtract_numeric', 'cum_sum', 'month',
                        'add_numeric', 'minute', 'multiply_numeric_scalar', 'not_equal', 'hour', 'and',
                        'greater_than', 'equal', 'scalar_subtract_numeric_feature', 'weekday',
                        'modulo_numeric_scalar', 'or', 'less_than_equal_to', 'longitude', 'absolute', 'haversine',
                        'divide_numeric_scalar', 'greater_than_equal_to_scalar', 'percentile', 'not', 'cum_min',
                        'negate', 'time_since', 'cum_count', 'isin', 'num_characters', 'add_numeric_scalar', 'year',
                        'week', 'divide_numeric', 'not_equal_scalar', 'second', 'greater_than_scalar',
                        'multiply_numeric', 'equal_scalar', 'day', 'modulo_by_feature', 'subtract_numeric_scalar',
                        'less_than_equal_to_scalar', 'divide_by_feature', 'latitude', 'less_than',
                        'less_than_scalar', 'modulo_numeric', 'time_since_previous', 'cum_max', 'is_null',
                        'greater_than_equal_to']
    ignore_variables = {}
    # Sidestepping featuretools bugs
    if db_name == 'kddcup2014':
        agg_primitives.remove('n_most_common')
        trans_primitives = None
    elif db_name == 'homecreditdefaultrisk':
        agg_primitives = ['median', 'mode', 'all', 'mean', 'percent_true', 'any', 'sum', 'num_true', 'last',
                          'count', 'num_unique', 'trend', 'std', 'max', 'skew', 'min']
        # trans_primitives = ['diff', 'subtract_numeric',
        #                         'add_numeric', 'and', 'greater_than', 'or',
        #                         'less_than_equal_to',
        #                         'percentile', 'not', 'negate',
        #                         'divide_numeric',
        #                         'multiply_numeric',
        #                         'modulo_by_feature',
        #                         'divide_by_feature', 'less_than',
        #                         'is_null', 'greater_than_equal_to']
        trans_primitives = None
    dfs_features = run_dfs(es, target_entity, agg_primitives, trans_primitives, ignore_variables, max_depth, n_jobs,
                           chunk_size)
    dfs_feat_file = os.path.join(data_dir, f'tabular_features_dfs_larger_{db_name}.pkl')
    dfs_features.to_pickle(dfs_feat_file)


if __name__ == '__main__':
    """
    For starting up databases from command line (python -m data.utils <db_name>)
    """
    db_name = sys.argv[1]
    get_db_container(db_name)
