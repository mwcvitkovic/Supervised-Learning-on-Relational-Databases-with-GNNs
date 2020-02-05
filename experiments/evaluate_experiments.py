import glob
import os
import subprocess

from __init__ import username, project_root
from start_evaluating import main as evaluate_main

if __name__ == '__main__':
    device = 'cuda'  # todo: change?
    num_workers = 8
    expt_root_dir = os.path.join(project_root, 'runs', 'GNN')

    while True:
        print(
            f"\n\n*****\n\nWARNING: Make sure you've chown'd the expt_root_dir {expt_root_dir} and dirs under it\n\n*****")
        for db_dir in sorted(glob.glob(os.path.join(expt_root_dir, '*'))):
            for model_dir in glob.glob(os.path.join(db_dir, '*')):
                for expt_dir in glob.glob(os.path.join(model_dir, '*')):
                    for model_logdir in glob.glob(os.path.join(expt_dir, '*')):
                        if 'LightGBM' in model_logdir:
                            continue
                        if model_logdir == 'homecreditdefaultrisk_dfs_smaller':
                            continue
                        for checkpoint_id in ['best_auroc', 'best_acc', 'best_loss']:
                            eval_path = os.path.join(model_logdir, 'evaluations', f'model_checkpoint_{checkpoint_id}')
                            if os.path.exists(eval_path) and (
                                    'results.json' in os.listdir(eval_path) or 'kaggle_submission.csv' in os.listdir(
                                eval_path)):
                                print(f'\n\nAlready evaluated {model_logdir} at checkpoint {checkpoint_id}.  Skipping')
                            elif not (os.path.exists(os.path.join(model_logdir, 'stopped_early.info')) or \
                                      os.path.exists(os.path.join(model_logdir, 'finished_all_epochs.info'))):
                                print(f'\n\nTraining not complete for {model_logdir}.  Skipping.')
                            else:
                                print(f'\n\nEvaluating {model_logdir} at checkpoint {checkpoint_id}')
                                subprocess.run(f'sudo chown -R {username} {model_logdir}', shell=True)
                                try:
                                    evaluate_main(dict(
                                        do_evaluate=True,
                                        do_dump_activations=False,
                                        module_acts_to_dump="",
                                        model_logdir=model_logdir,
                                        checkpoint_id=checkpoint_id,
                                        device=device,
                                        num_workers=num_workers))
                                except Exception as e:
                                    print(f'Exception while evaluating {model_logdir} at checkpoint {checkpoint_id}')
                                    print(e)
