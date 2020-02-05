import math
import os
from datetime import datetime

from data.utils import get_ds_info
from experiments.utils import run_script_with_kwargs

model_class_name = 'TabLogReg'

from experiments.tabular import ds_names


def get_kwargs(ds_name):
    ds_info = get_ds_info(ds_name)
    n_datapoints = ds_info['meta']['n_datapoints']
    n_columns = len(ds_info['meta']['columns'])

    weight_decay = 0.01

    ######################
    # Basic kwargs
    epochs = 500
    max_batch = 1024
    batch_size = min(n_datapoints // 30, max_batch)
    if batch_size == max_batch:
        batch_size // max(1, math.log10(n_columns) - 1)
    kwargs = dict(
        seed=1234,
        debug_network=False,
        encoders=dict(
            CATEGORICAL='CategoricalOrdinalEnc',
            SCALAR='ScalarRobustScalerEnc',
            DATETIME='DatetimeScalarEnc',
            LATLONG='LatLongScalarEnc',
            TEXT='TextSummaryScalarEnc'),
        early_stopping_patience=epochs,
        early_stopping_metric='loss',
        max_nodes_per_graph=False,
        train_fraction_to_use=1.0,
        dataset_name=ds_name,
        device='cuda',
        find_lr=False,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=8
    )
    # LR Schedule
    kwargs.update(
        lr_scheduler_class_name='StepLR',
        lr_scheduler_kwargs=dict(
            step_size=1,
            gamma=1.0
        ),
    )
    # Optimizer
    kwargs.update(
        optimizer_class_name='AdamW',
        optimizer_kwargs=dict(
            lr=5e-4,
            weight_decay=weight_decay,
        ),
        wd_bias=False,
        wd_embed=False,
        wd_bn=False,
    )
    # Sampler
    sampler_class_name = 'RandomSampler'
    sampler_class_kwargs = {}
    kwargs.update(sampler_class_name=sampler_class_name,
                  sampler_class_kwargs=sampler_class_kwargs)
    # Model
    kwargs.update(
        model_class_name=model_class_name,
        model_kwargs=dict()
    )
    kwargs['model_kwargs'].update(
        max_emb_dim=32,
        p_dropout=0.0,
        one_hot_embeddings=True,
        drop_whole_embeddings=False,
        activation_class_name='SELU',
        activation_class_kwargs={},
        norm_class_name='BatchNorm1d',
        norm_class_kwargs={},
        loss_class_name='CrossEntropyLoss',
        loss_class_kwargs={}
    )

    return kwargs


if __name__ == '__main__':
    for ds_name in ds_names:
        experiment_slug = datetime.now().strftime('%b%d_%H-%M-%S-%f')
        for train_test_split in [
            'xval0',
            'xval1',
            'xval2',
            'xval3',
            'xval4',
        ]:
            kwargs = get_kwargs(ds_name)
            kwargs['log_dir'] = os.path.join('deep_tabular',
                                             ds_name,
                                             model_class_name,
                                             experiment_slug,
                                             train_test_split)
            kwargs['train_test_split'] = train_test_split
            session_name = '_'.join(
                [ds_name, model_class_name, experiment_slug,
                 train_test_split])
            run_script_with_kwargs('start_training',
                                   kwargs,
                                   session_name,
                                   locale='local_tmux',
                                   n_gpu=1,
                                   n_cpu=kwargs['num_workers'],
                                   mb_memory=60000)  # this is the memory on a p3.2xlarge
