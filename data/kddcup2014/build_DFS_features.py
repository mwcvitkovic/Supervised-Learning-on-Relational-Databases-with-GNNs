import os

import featuretools as ft
import pandas as pd
from featuretools.variable_types import Text, LatLong, Categorical

from __init__ import data_root
from data.utils import get_db_info, set_entity_variable_types, compute_and_save_dfs_features

db_name = 'kddcup2014'
dp_limit = None
max_depth = 2
n_jobs = 1
chunk_size = 5000

if __name__ == '__main__':
    print('Loading data')
    data_dir = os.path.join(data_root, 'raw_data', db_name)
    db_info = get_db_info(db_name)

    projects = pd.read_csv(os.path.join(data_dir, 'projects.csv'), nrows=dp_limit, index_col=False)
    essays = pd.read_csv(os.path.join(data_dir, 'essays.csv'), nrows=dp_limit, index_col=False)
    resources = pd.read_csv(os.path.join(data_dir, 'resources.csv'), nrows=dp_limit, index_col=False)
    outcomes = pd.read_csv(os.path.join(data_dir, 'outcomes.csv'), usecols=['projectid', 'is_exciting'], nrows=dp_limit,
                           index_col=False)

    # Add target to projects and rename
    projects = projects.merge(outcomes, how='outer')
    projects.rename(columns={'is_exciting': 'TARGET'}, inplace=True)
    mapping = {'t': 1.0, 'f': 0.0}
    projects['TARGET'] = projects['TARGET'].replace(mapping)

    # Fix latlong
    projects['school_latlong'] = projects[['school_latitude', 'school_longitude']].apply(tuple, axis=1)
    projects.drop(['school_latitude', 'school_longitude'], axis=1, inplace=True)

    # Drop redundant column
    projects.drop('teacher_acctid', axis=1, inplace=True)
    essays.drop('teacher_acctid', axis=1, inplace=True)

    # Make entity set
    es = ft.EntitySet(id='projects')

    # Entities with a unique index
    es = es.entity_from_dataframe(entity_id='project',
                                  dataframe=projects,
                                  index='projectid',
                                  variable_types={'resource_type': Categorical,
                                                  'TARGET': Categorical,
                                                  'school_latlong': LatLong})
    es = es.entity_from_dataframe(entity_id='resource',
                                  dataframe=resources,
                                  index='resourceid',
                                  variable_types={'vendorid': Categorical,
                                                  'project_resource_type': Categorical,
                                                  'item_name': Text,  # setting here due to featuretools bug
                                                  'item_number': Text})
    # Entities that do not have a unique index
    es = es.entity_from_dataframe(entity_id='essay',
                                  dataframe=essays,
                                  make_index=True,
                                  index='essayid',
                                  variable_types={'title': Text,
                                                  'short_description': Text,
                                                  'need_statement': Text,
                                                  'essay': Text})

    # Define relationships (except featuretools can't handle multiple paths from a parent to a child)
    r_essay_project = ft.Relationship(es['project']['projectid'], es['essay']['projectid'])
    r_resource_project = ft.Relationship(es['project']['projectid'], es['resource']['projectid'])

    # Add in the defined relationships
    es = es.add_relationships([r_essay_project, r_resource_project])

    # Fix up variable types
    set_entity_variable_types(es['project'], 'Project', db_info)
    set_entity_variable_types(es['essay'], 'Essay', db_info)
    set_entity_variable_types(es['resource'], 'Resource', db_info)

    compute_and_save_dfs_features(es, 'project', db_name, max_depth, n_jobs, chunk_size)
