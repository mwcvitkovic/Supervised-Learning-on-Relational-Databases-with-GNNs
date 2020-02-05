from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns

from utils import get_train_val_test_datasets

sns.set(style="white", color_codes=True)
from tqdm import tqdm

from data.utils import get_db_container

db_names = (
    'acquirevaluedshopperschallenge',
    'homecreditdefaultrisk',
    'kddcup2014')

if __name__ == '__main__':
    while True:
        inp = input('Re-extract dataset info? (y/n): ')
        if inp in ['y', 'n']:
            break

    for db_name in db_names:
        print(f'Doing {db_name}')
        _ = get_db_container(db_name)
        train_dataset, val_dataset, test_dataset = get_train_val_test_datasets(db_name, 'use_full_train')
        datasets = {'train': train_dataset,
                    'val': val_dataset,
                    'test': test_dataset}
        df_graph_info_path = f'./experiments/{db_name}_df_graph_info.pkl'
        df_node_info_path = f'./experiments/{db_name}_df_node_info.pkl'

        if inp == 'y':
            n_nodes = []
            n_edges = []
            n_in_edges = []
            n_out_edges = []
            for split, dataset in datasets.items():
                for dp_id, (edge_list, node_types, edge_types, features, label) in tqdm(dataset):
                    n_nodes.append(len(node_types))
                    n_edges.append(2 * len(edge_types) + len(node_types))

                    if not node_types:
                        raise Exception
                    if not edge_list:
                        n_in_edges += [0] * len(node_types)
                        n_out_edges += [0] * len(node_types)
                    else:
                        edges = np.array(edge_list)
                        in_edge_count = Counter(edges[:, 1])
                        in_edge_count.update({n: 0 for n in range(len(node_types))})
                        in_edge_count = [in_edge_count[i] for i in range(len(node_types))]
                        n_in_edges += in_edge_count
                        out_edge_count = Counter(edges[:, 0])
                        out_edge_count.update({n: 0 for n in range(len(node_types))})
                        out_edge_count = [out_edge_count[i] for i in range(len(node_types))]
                        n_out_edges += out_edge_count

            df_graph_info = pd.DataFrame({'Number of nodes': n_nodes, 'Number of edges': n_edges})
            df_graph_info.to_pickle(df_graph_info_path)

            df_node_info = pd.DataFrame({'In Degree': n_in_edges, 'Out Degree': n_out_edges})
            df_node_info.to_pickle(df_node_info_path)

        df_graph_info = pd.read_pickle(df_graph_info_path)
        df_node_info = pd.read_pickle(df_node_info_path)

        df_graph_info = np.log10(df_graph_info)
        df_graph_info.columns = ('$\log_{10}$ number of nodes', '$\log_{10}$ number of edges')
        df_node_info = np.log10(df_node_info + 1)
        df_node_info.columns = ('$\log_{10}$ In Degree (including self-loops)',
                                '$\log_{10}$ Out Degree (including self-loops)')

        g = sns.jointplot(x=df_graph_info.columns[0],
                          y=df_graph_info.columns[1],
                          data=df_graph_info,
                          kind='hex')
        g.savefig(f'./experiments/{db_name}_n_node_vs_n_edge.svg', format='svg')
        #
        # g = sns.jointplot(x=df_node_info.columns[0],
        #                   y=df_node_info.columns[1],
        #                   data=df_node_info,
        #                   kind='scatter')
        # g.savefig(f'./experiments/{db_name}_in_degree_vs_out_degree.svg', format='svg')
