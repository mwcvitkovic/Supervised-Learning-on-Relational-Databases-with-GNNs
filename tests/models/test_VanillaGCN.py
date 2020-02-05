import unittest

import torch
from torch.optim import AdamW
from tqdm import tqdm

from data.utils import get_db_info
from models import GCN
from models.utils import recursive_to
from utils import get_train_val_test_datasets, get_dataloader, DummyWriter


class TestGCN(unittest.TestCase):
    db_names = (
        'acquirevaluedshopperschallenge',
        'homecreditdefaultrisk',
        'kddcup2014',
    )

    def test_memorize_minibatch(self):
        for db_name in self.db_names:
            db_info = get_db_info(db_name)
            train_data, val_data, _ = get_train_val_test_datasets(dataset_name=db_name,
                                                                  train_test_split='use_full_train',
                                                                  encoders=dict(
                                                                      CATEGORICAL='CategoricalOrdinalEnc',
                                                                      SCALAR='ScalarRobustScalerEnc',
                                                                      DATETIME='DatetimeScalarEnc',
                                                                      LATLONG='LatLongScalarEnc',
                                                                      TEXT='TextSummaryScalarEnc'), )
            train_loader = get_dataloader(dataset=train_data,
                                          batch_size=256,
                                          sampler_class_name='SequentialSampler',
                                          num_workers=0,
                                          max_nodes_per_graph=False)

            writer = DummyWriter()
            model = GCN(writer, db_info=db_info, hidden_dim=256, n_init_layers=3, activation_class_name='SELU',
                        activation_class_kwargs={}, loss_class_kwargs={}, loss_class_name='CrossEntropyLoss',
                        p_dropout=0.0, drop_whole_embeddings=True, n_layers=3, readout_class_name='AvgPooling',
                        readout_kwargs={})
            if torch.cuda.is_available():
                model.cuda()
                model.device = torch.device('cuda:0')
            else:
                model.device = torch.device('cpu')
            model.train()
            optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.0)

            bdgl, features, label = next(iter(train_loader))
            recursive_to((bdgl, features, label), model.device)
            for _ in tqdm(range(200)):
                optimizer.zero_grad()
                output = model(bdgl, features)
                loss = model.loss_fxn(output, label)
                if loss < 1e-4:
                    break
                loss.backward()
                optimizer.step()
            else:
                tqdm.write(f'Loss: {loss}')
                self.fail("Didn't memorize minibatch")
