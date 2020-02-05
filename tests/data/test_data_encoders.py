import unittest

import torch
from tqdm import tqdm

from data.utils import get_db_container, get_db_info
from utils import get_dataloader, get_train_val_test_datasets

dataset_names = (
    'acquirevaluedshopperschallenge',
    'homecreditdefaultrisk',
    'kddcup2014',
)

scalar_encoders = (
    'ScalarRobustScalerEnc',
    'ScalarPowerTransformerEnc',
    'ScalarQuantileTransformerEnc',
)


class TestDataEncoders(unittest.TestCase):
    def get_loaders(self, db_name, encoders, batch_size, num_workers):
        db_info = get_db_info(db_name)
        max_nodes_per_graph = None
        _ = get_db_container(db_name)
        train_data, val_data, test_data = get_train_val_test_datasets(dataset_name=db_name,
                                                                      train_test_split='use_full_train',
                                                                      encoders=encoders)
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
        loaders = {'train': train_loader,
                   'val': val_loader,
                   'test': test_loader}
        return db_info, loaders

    def test_datapoints_for_appropriate_null_flags_for_scalar_encoders(self):
        for db_name in dataset_names:
            for scalar_encoder in scalar_encoders:
                encoders = {'SCALAR': scalar_encoder}
                db_info, loaders = self.get_loaders(db_name, encoders, batch_size=512, num_workers=0)
                for split, loader in loaders.items():
                    for bdgl, features, label in tqdm(loader):
                        for node_type, node_features in features.items():
                            for feature_name, feature_data in node_features.items():
                                feature_type = db_info['node_types_and_features'][node_type][feature_name]['type']
                                if feature_type == 'SCALAR':
                                    self.assertEqual((feature_data.sum(dim=1) == 0).sum().item(), 0,
                                                     "Something didn't get initialized correctly")
                                    supposedly_null_values = feature_data[torch.where(feature_data[:, 1] == 1)][:, 0]
                                    self.assertEqual((supposedly_null_values != 0).sum().item(), 0)

    def test_datapoints_have_categorical_value_zero_only_when_they_are_None_in_the_raw_data(self):
        for db_name in dataset_names:
            for scalar_encoder in scalar_encoders:
                encoders = {'SCALAR': scalar_encoder}
                db_info, loaders = self.get_loaders(db_name, encoders, batch_size=1, num_workers=0)
                for split, loader in loaders.items():
                    dataset = loader.dataset
                    for bdgl, features, label in tqdm(loader):
                        _, (_, _, _, raw_dp_feats, _) = dataset.get_dp_by_id(bdgl.dp_ids[0])
                        for node_type, node_features in features.items():
                            for feature_name, feature_data in node_features.items():
                                feature_type = db_info['node_types_and_features'][node_type][feature_name]['type']
                                if feature_type == 'CATEGORICAL':
                                    for idx in torch.where(feature_data == 0)[0]:
                                        dp_feat = raw_dp_feats[node_type][feature_name][idx]
                                        self.assertIsNone(dp_feat)
