import os
import pickle

import numpy as np
from torch.utils.data import Dataset

from __init__ import data_root
from data.data_encoders import LatLongScalarEnc, DatetimeScalarEnc, CategoricalOrdinalEnc, ScalarRobustScalerEnc, \
    TfidfEnc, ScalarPowerTransformerEnc, \
    ScalarQuantileTransformerEnc, TextSummaryScalarEnc, ScalarQuantileOrdinalEnc
from data.utils import get_db_info


class DatabaseDataset(Dataset):
    def __init__(self, db_name=None, datapoint_ids=None, encoders=None):
        self.db_name = db_name
        self.datapoint_ids = datapoint_ids
        self.data_dir = os.path.join(data_root, self.db_name, 'preprocessed_datapoints')
        os.makedirs(self.data_dir, exist_ok=True)

        self.db_info = get_db_info(self.db_name)

        # Download data if necessary
        all_dps_present = len(os.listdir(self.data_dir)) == self.db_info['task']['n_train'] + self.db_info['task'][
            'n_test']
        assert all_dps_present

        self.feature_encoders = {}
        for node_type, features in self.db_info['node_types_and_features'].items():
            self.feature_encoders[node_type] = dict()
            for feature_name, feature_info in features.items():
                if feature_info['type'] == 'CATEGORICAL':
                    enc = CategoricalOrdinalEnc(feature_info['sorted_values'])
                elif feature_info['type'] == 'SCALAR':
                    s_enc = encoders['SCALAR']
                    if s_enc == 'ScalarRobustScalerEnc':
                        enc = ScalarRobustScalerEnc(feature_info['RobustScaler_center_'],
                                                    feature_info['RobustScaler_scale_'])
                    elif s_enc == 'ScalarPowerTransformerEnc':
                        enc = ScalarPowerTransformerEnc(feature_info['PowerTransformer_lambdas_'],
                                                        feature_info['PowerTransformer_scale_'],
                                                        feature_info['PowerTransformer_mean_'],
                                                        feature_info['PowerTransformer_var_'],
                                                        feature_info['PowerTransformer_n_samples_seen_'])
                    elif s_enc == 'ScalarQuantileTransformerEnc':
                        enc = ScalarQuantileTransformerEnc(feature_info['QuantileTransformer_n_quantiles_'],
                                                           feature_info['QuantileTransformer_quantiles_'],
                                                           feature_info['QuantileTransformer_references_'])
                    elif s_enc == 'ScalarQuantileOrdinalEnc':
                        enc = ScalarQuantileOrdinalEnc(feature_info['KBinsDiscretizer_n_bins_'],
                                                       feature_info['KBinsDiscretizer_bin_edges_'])
                    else:
                        raise ValueError(f'scalar encoder {s_enc} not recognized')
                elif feature_info['type'] == 'DATETIME':
                    enc = DatetimeScalarEnc()
                elif feature_info['type'] == 'LATLONG':
                    enc = LatLongScalarEnc()
                elif feature_info['type'] == 'TEXT':
                    t_enc = encoders['TEXT']
                    if t_enc == 'TfidfEnc':
                        enc = TfidfEnc(feature_info['Tfidf_vocabulary_'],
                                       feature_info['Tfidf_idf_'])
                    elif t_enc == 'TextSummaryScalarEnc':
                        enc = TextSummaryScalarEnc(feature_info['RobustScaler_center_'],
                                                   feature_info['RobustScaler_scale_'])
                self.feature_encoders[node_type][feature_name] = enc

    def __len__(self):
        return len(self.datapoint_ids)

    def __getitem__(self, item: int):
        dp_id = self.datapoint_ids[item]
        with open(os.path.join(self.data_dir, str(dp_id)), 'rb') as f:
            dp = pickle.load(f)
        return dp_id, dp

    def get_dp_by_id(self, dp_id):
        idx = np.argwhere(self.datapoint_ids == dp_id).item()
        return self[idx]

    def write_kaggle_submission_file(self, outputs, path):
        raise NotImplementedError
