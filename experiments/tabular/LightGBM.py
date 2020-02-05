import os
from datetime import datetime

from experiments.utils import run_script_with_kwargs

model_class_name = 'LightGBM'

from experiments.tabular import ds_names

kwargs = dict(
    seed=987,
    debug_network=False,
    encoders=dict(
        CATEGORICAL='CategoricalOrdinalEnc',
        SCALAR='ScalarRobustScalerEnc',
        DATETIME='DatetimeScalarEnc',
        LATLONG='LatLongScalarEnc',
        TEXT='TextSummaryScalarEnc'),
    train_fraction_to_use=1.0,
    model_class_name='LightGBM',
    model_kwargs=dict(
        num_leaves=30,
        min_data_in_leaf=20,
    ),
    num_boost_round=300,
    early_stopping_patience=20,
    num_workers=1,
)

if __name__ == '__main__':
    for ds_name in ds_names:
        experiment_slug = datetime.now().strftime('%b%d_%H-%M-%S-%f')
        for train_test_split in [
            'use_full_train',
            'xval0',
            'xval1',
            'xval2',
            'xval3',
            'xval4',
        ]:
            kwargs['log_dir'] = os.path.join('deep_tabular',
                                             ds_name,
                                             model_class_name,
                                             experiment_slug,
                                             train_test_split)
            kwargs['train_test_split'] = train_test_split
            kwargs['dataset_name'] = ds_name
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
