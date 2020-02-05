import glob
import json
import os
import subprocess

import numpy as np
import pandas as pd

from __init__ import username, project_root
from data.TabularDataset import TabularDataset
from data.utils import get_ds_info
from start_evaluating import main as evaluate_main
from utils import get_train_test_dp_ids

module_acts_to_dump = {
    'PoolMLP': "fcout[-1]",
    'GCN': "fcout[-1]",
    'GIN': "fcout[-1]",
    'GAT': "fcout[-1]",
}

expt_root_dir = os.path.join(project_root, 'runs', 'GNN')
checkpoint_id = 'best_auroc'

device = 'cuda:0'
num_workers = 8

if __name__ == '__main__':
    print(
        f"\n\n*****\n\nWARNING: Make sure you've chown'd the expt_root_dir {expt_root_dir} and dirs under it\n\n*****")
    for db_name in sorted(os.listdir(os.path.join(expt_root_dir))):
        db_dir = os.path.join(expt_root_dir, db_name)
        for model_name in sorted(os.listdir(db_dir)):
            model_dir = os.path.join(db_dir, model_name)
            matd = module_acts_to_dump[model_name]
            for expt_dir in glob.glob(os.path.join(model_dir, '*')):
                for train_test_split in os.listdir(os.path.join(expt_dir)):
                    logdir = os.path.join(expt_dir, train_test_split)
                    chkpt_dir = os.path.join(logdir, 'evaluations', f'model_checkpoint_{checkpoint_id}')
                    acts_file = os.path.join(chkpt_dir, f'{matd}.activations.npy')
                    acts = None
                    if os.path.exists(acts_file):
                        print(f'\n\nAlready dumped acts {acts_file}.  Not dumping again.')
                    elif not (os.path.exists(os.path.join(logdir, 'stopped_early.info')) or \
                              os.path.exists(os.path.join(logdir, 'finished_all_epochs.info'))):
                        print(f'\n\nTraining not complete for {logdir}.  Skipping.')
                        continue
                    else:
                        print(f'\n\nDumping acts for {logdir} at checkpoint {checkpoint_id}')
                        subprocess.run(f'sudo chown -R {username} {logdir}', shell=True)
                        try:
                            acts = evaluate_main(dict(
                                do_evaluate=False,
                                do_dump_activations=True,
                                module_acts_to_dump=matd,
                                model_logdir=logdir,
                                checkpoint_id=checkpoint_id,
                                device=device,
                                num_workers=num_workers))
                        except Exception as e:
                            print(f'Exception while evaluating {logdir} at checkpoint {checkpoint_id}')
                            print(e.__repr__())
                            continue

                    if acts is None:
                        acts = np.load(acts_file)
                        acts = pd.DataFrame(acts)

                    new_ds_name = f'{db_name}_{model_name}_{matd}_{train_test_split}'
                    new_ds_file = os.path.join(chkpt_dir, new_ds_name + '.csv.gz')
                    if os.path.exists(new_ds_file):
                        print(f'Already made new dataset.  Moving on.')
                    else:
                        print(f'Appending acts to original dataset and saving to {new_ds_file}')
                        orig_dataset_name = f'{db_name}_main_table'
                        train_dp_ids, test_dp_ids = get_train_test_dp_ids(orig_dataset_name)
                        dp_ids = np.concatenate(
                            [train_dp_ids, test_dp_ids]) if test_dp_ids is not None else train_dp_ids
                        orig_dataset = TabularDataset(orig_dataset_name, dp_ids, encoders=None)
                        orig_data = orig_dataset.raw_data
                        orig_dataset_ds_info = get_ds_info(orig_dataset_name)
                        orig_dataset_ds_info['processed']['local_path'] = new_ds_file

                        acts = acts.set_index(orig_data.index)
                        act_cols = [f'{model_name}_act{i}' for i in acts.columns]
                        acts = acts.rename(columns={i: f'{model_name}_act{i}' for i in acts.columns})
                        targets = orig_dataset.targets.numpy()
                        targets = pd.DataFrame({'TARGET': [i if i in [0, 1] else np.nan for i in targets]})
                        targets = targets.set_index(orig_data.index)
                        new_dataset = pd.concat([orig_data, acts, targets], axis=1)
                        new_dataset.to_csv(new_ds_file)

                        orig_dataset_ds_info['meta']['name'] = new_ds_name
                        orig_dataset_ds_info['meta']['columns'] += [{'name': n, 'type': 'SCALAR'} for n in act_cols]
                        new_ds_info_file = os.path.join(chkpt_dir, f'{new_ds_name}.ds_info.json')
                        with open(new_ds_info_file, 'w') as f:
                            json.dump(orig_dataset_ds_info, f, indent=2)
