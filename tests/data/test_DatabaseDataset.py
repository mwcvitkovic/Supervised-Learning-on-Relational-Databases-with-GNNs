import unittest
import warnings

import torch
from tqdm import tqdm

from data.utils import get_db_container, get_db_info
from utils import get_dataloader, get_train_val_test_datasets

dataset_names = ('acquirevaluedshopperschallenge',
                 'homecreditdefaultrisk',
                 'kddcup2014')


class TestDatabaseDataset(unittest.TestCase):
    def test_datapoints_for_no_self_loops_and_nonnegative_edge_types(self):
        for db_name in dataset_names:
            for dataset in get_train_val_test_datasets(dataset_name=db_name,
                                                       train_test_split='use_full_train',
                                                       encoders=dict(
                                                           CATEGORICAL='CategoricalOrdinalEnc',
                                                           SCALAR='ScalarRobustScalerEnc',
                                                           DATETIME='DatetimeScalarEnc',
                                                           LATLONG='LatLongScalarEnc',
                                                           TEXT='TextSummaryScalarEnc'), ):
                for dp_id, (edge_list, node_types, edge_types, features, label) in tqdm(dataset):
                    # Nodes don't have any self loops in the raw data
                    for edge in edge_list:
                        self.assertNotEqual(edge[0], edge[1])

                    # All edge types are nonnegative in the raw data
                    self.assertTrue(all(et >= 0 for et in edge_types))

    def test_train_val_and_test_splits_contain_different_datapoints(self):
        for train_test_split in ['use_full_train', 'xval0', 'xval1', 'xval2', 'xval3', 'xval4']:
            for db_name in dataset_names:
                train_data, val_data, test_data = get_train_val_test_datasets(dataset_name=db_name,
                                                                              train_test_split=train_test_split,
                                                                              encoders=dict(
                                                                                  CATEGORICAL='CategoricalOrdinalEnc',
                                                                                  SCALAR='ScalarRobustScalerEnc',
                                                                                  DATETIME='DatetimeScalarEnc',
                                                                                  LATLONG='LatLongScalarEnc',
                                                                                  TEXT='TextSummaryScalarEnc'), )
                self.assertEqual(0, len(
                    set(train_data.datapoint_ids).intersection(val_data.datapoint_ids).intersection(
                        test_data.datapoint_ids)))


class TestDataBaseClass:
    class TestData(unittest.TestCase):
        db_name = None

        def setUp(self):
            self.db_info = get_db_info(self.db_name)
            batch_size = 1
            num_workers = 0
            max_nodes_per_graph = 100000
            _ = get_db_container(self.db_name)
            train_data, val_data, test_data = get_train_val_test_datasets(dataset_name=self.db_name,
                                                                          train_test_split='use_full_train',
                                                                          encoders=dict(
                                                                              CATEGORICAL='CategoricalOrdinalEnc',
                                                                              SCALAR='ScalarRobustScalerEnc',
                                                                              DATETIME='DatetimeScalarEnc',
                                                                              LATLONG='LatLongScalarEnc',
                                                                              TEXT='TextSummaryScalarEnc'), )
            train_loader = get_dataloader(dataset=train_data,
                                          batch_size=batch_size,
                                          sampler_class_name='SequentialSampler',
                                          num_workers=num_workers,
                                          max_nodes_per_graph=max_nodes_per_graph)
            val_loader = get_dataloader(dataset=val_data,
                                        batch_size=batch_size,
                                        sampler_class_name='SequentialSampler',
                                        num_workers=num_workers,
                                        max_nodes_per_graph=max_nodes_per_graph)
            test_loader = get_dataloader(dataset=test_data,
                                         batch_size=batch_size,
                                         sampler_class_name='SequentialSampler',
                                         num_workers=num_workers,
                                         max_nodes_per_graph=max_nodes_per_graph)
            self.loaders = {'train': train_loader,
                            'val': val_loader,
                            'test': test_loader}

        def test_loaded_datapoints(self):
            label_node_type, label_feature_name = self.db_info['label_feature'].split('.')
            for split, loader in self.loaders.items():
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    for bdgl, features, label in tqdm(loader):
                        # No empty graphs
                        self.assertGreater(bdgl.number_of_nodes(), 0)

                        # Every edge has an equal and opposite edge with negative edge type
                        uvt = torch.stack((*bdgl.all_edges('uv', 'eid'), bdgl.edata['edge_types'])).t()
                        u_v_type = []
                        v_u_negtype = []
                        for u, v, type in uvt.tolist():
                            u_v_type.append((u, v, type))
                            v_u_negtype.append((v, u, -type))
                        u_v_type_set = set(u_v_type)
                        v_u_negtype_set = set(v_u_negtype)
                        self.assertEqual(uvt.shape[0], len(u_v_type_set))  # Make sure no redundant edges
                        self.assertEqual(u_v_type_set, v_u_negtype_set)

                        # Every node gets a self loop after collation
                        for i in range(bdgl.number_of_nodes()):
                            self.assertIn((i, i, 0), u_v_type_set)

                        # Self loops have type 0
                        for u, v, type in u_v_type:
                            if u == v:
                                self.assertEqual(0, type)

                        # Features have all the right keys and numbers of values
                        self.assertGreater(len(features.keys()), 0)
                        for node_type, feats in features.items():
                            feat_keys = set(feats.keys())
                            # Ignore the label feature
                            if node_type == label_node_type:
                                feat_keys = feat_keys.union([label_feature_name])
                            self.assertEqual(feat_keys, self.db_info['node_types_and_features'][node_type].keys())
                            node_type_int = self.db_info['node_type_to_int'][node_type]
                            n_nodes_this_type = (bdgl.ndata['node_types'] == node_type_int).sum().item()
                            for feat_vals in feats.values():
                                self.assertEqual(n_nodes_this_type, feat_vals.shape[0])

                        # Only test points have labels
                        if split == 'test':
                            self.assertIsNone(label)
                        else:
                            self.assertIsNotNone(label)

                        # Label isn't present in features
                        self.assertNotIn(label_feature_name, features[label_node_type].keys())

        def test_null_counts_in_database_are_reasonable_and_match_preprocessed_datapoints(self):
            # Count up nulls in preprocessed datapoints
            n_null_counts = {}
            for split, loader in self.loaders.items():
                for _, (_, _, _, features, _) in tqdm(loader.dataset):
                    for node_type, f in features.items():
                        n_null_counts.setdefault(node_type, {})
                        for feature_name, values in f.items():
                            n_null_counts[node_type].setdefault(feature_name, 0)
                            n_null_counts[node_type][feature_name] += values.count(None)

            # Make sure nulls in preprocessed datapoints match those in db_info
            for node_type, features in self.db_info['node_types_and_features'].items():
                for feature_name, feature_info in features.items():
                    # Skip target feature, because it's not in the node features
                    if self.db_info['label_feature'] == '{}.{}'.format(node_type, feature_name):
                        continue
                    self.assertEqual(n_null_counts[node_type][feature_name], feature_info['n_null_values'],
                                     f'node_type: {node_type}, feature_name: {feature_name}')
