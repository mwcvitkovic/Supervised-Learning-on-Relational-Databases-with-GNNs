import os
from datetime import datetime

from __init__ import project_root
from experiments.tabular.dump_GNN_activations_and_build_datasets import module_acts_to_dump
from experiments.utils import run_script_with_kwargs

db_names = (
    'acquirevaluedshopperschallenge',
    'homecreditdefaultrisk',
    'kddcup2014',
)

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

gnn_expt_root_dir = os.path.join(project_root, 'runs', 'GNN')

if __name__ == '__main__':
    for db_name in db_names:
        experiment_slug = datetime.now().strftime('%b%d_%H-%M-%S-%f')
        for model_class_name, matd in module_acts_to_dump.items():
            tab_model_name = model_class_name + f'_{matd}_LightGBM'
            for train_test_split in [
                'use_full_train',
                'xval0',
                'xval1',
                'xval2',
                'xval3',
                'xval4',
            ]:
                kwargs['log_dir'] = os.path.join('GNN_stacked',
                                                 db_name,
                                                 tab_model_name,
                                                 experiment_slug,
                                                 train_test_split)
                kwargs['train_test_split'] = train_test_split
                model_dir = os.path.join(gnn_expt_root_dir, db_name, model_class_name)
                runs = os.listdir(model_dir)
                try:
                    assert len(runs) == 1, f'Multiple runs in {model_dir}'
                except Exception as e:
                    print(f'Error when finding {model_dir} pretraining run: {e}\n\n\n\n')
                    continue
                run_dir = os.path.join(model_dir, runs[0])
                ds_name = os.path.join(run_dir,
                                       train_test_split,
                                       'evaluations',
                                       'model_checkpoint_best_auroc',
                                       f'{db_name}_{model_class_name}_{matd}_{train_test_split}.ds_info.json')
                kwargs['dataset_name'] = ds_name
                session_name = '_'.join([db_name, model_class_name, experiment_slug, train_test_split])
                run_script_with_kwargs('start_training',
                                       kwargs,
                                       session_name,
                                       locale='local_tmux',
                                       n_gpu=1,
                                       n_cpu=kwargs['num_workers'],
                                       mb_memory=60000)  # this is the memory on a p3.2xlarge
