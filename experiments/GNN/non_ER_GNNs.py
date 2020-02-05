import os
from datetime import datetime

from experiments.utils import run_script_with_kwargs

model_class_names = (
    'PoolMLP',
    'GCN',
    'GAT',
    'GIN',
)

db_names = (
    'acquirevaluedshopperschallenge',
    'homecreditdefaultrisk',
    'kddcup2014',
)


def get_kwargs(model_class_name, db_name):
    # Basic kwargs
    epochs = 400
    kwargs = dict(
        seed=1234,
        debug_network=False,
        early_stopping_patience=20,
        early_stopping_metric='loss',
        train_fraction_to_use=1.0,
        device='cuda',
        find_lr=True,  # Careful if you change this to False - lots of options below assume it's true
        epochs=epochs,
    )
    # LR scheduler
    kwargs.update(
        lr_scheduler_class_name='ReduceLROnPlateau',
        lr_scheduler_kwargs=dict(
            mode='max',
            factor=0.1,
            patience=10,
            threshold=1e-4,
            threshold_mode='rel'
        )
    )
    # Sampler
    kwargs.update(sampler_class_name='RandomSampler',
                  sampler_class_kwargs={})
    # Optimizer
    kwargs.update(
        optimizer_class_name='AdamW',
        optimizer_kwargs=dict(
            lr=5e-6,
            weight_decay=0.0
        ),
        wd_bias=False,
        wd_embed=False,
        wd_bn=False,
    )

    ######################
    # Database-specific kwargs
    if db_name == 'acquirevaluedshopperschallenge':
        kwargs.update(
            dataset_name='acquirevaluedshopperschallenge',
            max_nodes_per_graph=50000,
            batch_size=128,
            num_workers=8
        )
    elif db_name == 'homecreditdefaultrisk':
        kwargs.update(
            dataset_name='homecreditdefaultrisk',
            max_nodes_per_graph=False,
            batch_size=512,
            num_workers=8
        )
    elif db_name == 'kddcup2014':
        kwargs.update(
            dataset_name='kddcup2014',
            max_nodes_per_graph=False,
            batch_size=512,
            num_workers=8
        )

    ######################
    # Model-specific kwargs
    if model_class_name == 'PoolMLP':
        kwargs.update(
            batch_size=kwargs['batch_size'] * 2,
            model_class_name=model_class_name,
            model_kwargs=dict(
                p_dropout=0.0,
                drop_whole_embeddings=True,
            )
        )
    elif model_class_name == 'GCN':
        kwargs.update(
            batch_size=kwargs['batch_size'] * 3 // 2,
            model_class_name=model_class_name,
            model_kwargs=dict(
                p_dropout=0.0,
                drop_whole_embeddings=True,
                n_layers=3,
            ),
        )
    elif model_class_name == 'GAT':
        kwargs.update(
            batch_size=kwargs['batch_size'] * 3 // 2,
            model_class_name=model_class_name,
            model_kwargs=dict(
                p_dropout=0.0,
                drop_whole_embeddings=True,
                n_layers=3,
                n_heads=4,
                residual=True
            )
        )
    elif model_class_name == 'GIN':
        kwargs.update(
            batch_size=kwargs['batch_size'] * 2,
            model_class_name=model_class_name,
            model_kwargs=dict(
                p_dropout=0.0,
                drop_whole_embeddings=True,
                n_layers=3,
                aggregator_type='sum',
                init_eps=0.001,
                learn_eps=True
            )
        )
    kwargs['model_kwargs'].update(
        hidden_dim=256,
        n_init_layers=3,
        activation_class_name='SELU',
        activation_class_kwargs={},
    )

    # Readout kwargs
    kwargs['model_kwargs'].update(
        readout_class_name='AvgPooling',
        readout_kwargs={},
    )
    # Loss function
    kwargs['model_kwargs'].update(
        loss_class_name='CrossEntropyLoss',
        loss_class_kwargs={}
    )

    return kwargs


if __name__ == '__main__':
    for model_class_name in model_class_names:
        for db_name in db_names:
            experiment_slug = datetime.now().strftime('%b%d_%H-%M-%S-%f')
            for train_test_split in ['use_full_train', 'xval0', 'xval1', 'xval2', 'xval3', 'xval4']:
                kwargs = get_kwargs(model_class_name, db_name)
                kwargs['log_dir'] = os.path.join('non_ER_GNN', db_name, model_class_name, experiment_slug,
                                                 train_test_split)
                kwargs['train_test_split'] = train_test_split
                session_name = '_'.join(['GNN', db_name, model_class_name, train_test_split])
                run_script_with_kwargs('start_training',
                                       kwargs,
                                       session_name,
                                       locale='local_tmux',
                                       n_gpu=1,
                                       n_cpu=kwargs['num_workers'],
                                       mb_memory=60000)  # this is the memory on a p3.2xlarge
