import json
import math
import os

import pandas as pd
import seaborn as sns

from __init__ import project_root

sns.set(style="whitegrid")

single_table_ds_names = (
    'acquirevaluedshopperschallenge_main_table',
    'homecreditdefaultrisk_main_table',
    'kddcup2014_main_table_and_essays',
)

dfs_table_ds_names = (
    'acquirevaluedshopperschallenge_dfs_larger',
    'homecreditdefaultrisk_dfs_smaller',
    'kddcup2014_dfs_larger',
)

tab_model_names = ('TabLogReg', 'TabMLP', 'LightGBM')

tab_expt_root_dir = os.path.join(project_root, 'runs', 'deep_tabular')

gnn_ds_names = (
    'acquirevaluedshopperschallenge',
    'homecreditdefaultrisk',
    'kddcup2014',
)

gnn_model_names = (
    'PoolMLP',
    'GCN',
    'GIN',
    'GAT',
)

gnn_expt_root_dir = os.path.join(project_root, 'runs', 'GNN')

gnn_stacked_model_names = (
    'PoolMLP_fcout[-1]_LightGBM',
    'GCN_fcout[-1]_LightGBM',
    'GIN_fcout[-1]_LightGBM',
    'GAT_fcout[-1]_LightGBM',
)

gnn_stacked_expt_root_dir = os.path.join(project_root, 'runs', 'GNN_stacked')


def fmt_row_for_latex(row):
    mean, std = row['value'], row['std']
    if std:
        n_zeros = -int(math.floor(math.log10(std)))
    else:
        n_zeros = 0
    std = round(std, n_zeros)
    mean = round(mean, n_zeros)
    return f'$ {mean} \pm {std} $'


if __name__ == '__main__':
    for ds_names, model_names, expt_root_dir in ((single_table_ds_names, tab_model_names, tab_expt_root_dir),
                                                 (dfs_table_ds_names, tab_model_names, tab_expt_root_dir),
                                                 (gnn_ds_names, gnn_model_names, gnn_expt_root_dir),
            # (gnn_ds_names, gnn_stacked_model_names, gnn_stacked_expt_root_dir),
                                                 ):
        results = []
        for chkpt_name, metric_name in [('best_acc', 'test_accuracy'), ('best_auroc', 'test_auroc')]:
            for ds_name in ds_names:
                ds_dir = os.path.join(expt_root_dir, ds_name)
                for model_name in sorted(os.listdir(ds_dir)):
                    model_dir = os.path.join(ds_dir, model_name)

                    if model_name not in model_names:
                        continue

                    runs = os.listdir(model_dir)
                    assert len(runs) == 1
                    run_dir = os.path.join(model_dir, runs[0])
                    for train_test_split in ['xval0', 'xval1', 'xval2', 'xval3', 'xval4']:
                        chkpts = [p for p in os.listdir(os.path.join(run_dir, train_test_split, 'evaluations')) if
                                  chkpt_name in p]
                        assert len(chkpts) == 1
                        chkpt = chkpts[0]
                        eval_results = os.path.join(run_dir, train_test_split, 'evaluations', chkpt, 'results.json')
                        result = {'model_name': model_name,
                                  'ds_name': ds_name,
                                  'train_test_split': train_test_split,
                                  'chkpt_name': chkpt_name}
                        with open(eval_results, 'r') as f:
                            metrics = json.load(f)
                        result.update(value=metrics[metric_name])
                        results.append(result)

        df = pd.DataFrame(results)
        for chkpt_name in ['best_acc', 'best_auroc']:
            cols = []
            for model_name in model_names:
                model_results = df.loc[(df['model_name'] == model_name) & (df['chkpt_name'] == chkpt_name)]
                agg_results = model_results.groupby('ds_name', as_index=True).agg({'value': 'mean'})
                agg_results['std'] = model_results.groupby('ds_name', as_index=True).agg({'value': 'std'})
                tab_results = agg_results.apply(fmt_row_for_latex, axis=1)
                tab_results = pd.DataFrame({model_name: tab_results})
                cols.append(tab_results)
            table_results = cols[0].join(cols[1:])
            print(f'\nResults for {chkpt_name}\n')
            print(table_results.transpose().reset_index().to_latex(escape=False, index=False,
                                                                   formatters={'ds_name': lambda x: x[:15]}).replace(
                '_',
                '\_'))

        best_accs = df.loc[df['chkpt_name'] == 'best_acc']
        g = sns.catplot(y='ds_name', x='value', hue='model_name', data=best_accs, kind='bar', height=10, capsize=0.2)
        g.savefig(f'tabular_results_best_acc.svg', format='svg')
        best_aurocs = df.loc[df['chkpt_name'] == 'best_auroc']
        g = sns.catplot(y='ds_name', x='value', hue='model_name', data=best_aurocs, kind='bar', height=10, capsize=0.2)
        g.savefig(f'tabular_results_best_auroc.svg', format='svg')
        # plt.show()
