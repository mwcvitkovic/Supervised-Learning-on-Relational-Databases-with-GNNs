import os
import pickle
import sys

import neotime

from data.utils import get_db_info, get_neo4j_db_driver


def create_datapoint_from_database(db_name, base_query, target_dir, dp_id):
    db_info = get_db_info(db_name)

    # Get graph from database
    driver = get_neo4j_db_driver(db_name)
    with driver.session() as session:
        query = base_query.format(dp_id)
        result = session.run(query)
        g = result.graph()

    # Construct DGLGraph for each neo4j graph, and batch them
    # Also collect the features and labels from each graph
    label_node_type, label_feature_name = db_info['label_feature'].split('.')
    features = {}
    for node_type in db_info['node_types_and_features'].keys():
        features[node_type] = {}
        for feature_name in db_info['node_types_and_features'][node_type].keys():
            # Making sure not to include the label value among the training features
            if not (node_type == label_node_type and feature_name == label_feature_name):
                features[node_type][feature_name] = []

    neo4j_id_to_graph_idx = {node.id: idx for idx, node in enumerate(g.nodes)}
    node_types = [None] * len(g.nodes)
    for node in g.nodes:
        node_type = tuple(node.labels)[0]
        node_idx = neo4j_id_to_graph_idx[node.id]
        node_types[node_idx] = db_info['node_type_to_int'][node_type]
        for feature_name, feature_values in features[node_type].items():
            # Dealing with latlongs
            if db_info['node_types_and_features'][node_type][feature_name]['type'] == 'LATLONG':
                lat_name, lon_name = feature_name.split('+++')
                value = (node.get(lat_name), node.get(lon_name))
            else:
                value = node.get(feature_name)
            # neotime doesn't pickle well
            if isinstance(value, (neotime.Date, neotime.DateTime)):
                value = value.to_native()
            feature_values.append(value)
        if node_type == label_node_type:
            label = node.get(label_feature_name)

    edge_list = []
    edge_types = []
    for rel in g.relationships:
        start_node_idx = neo4j_id_to_graph_idx[rel.start_node.id]
        end_node_idx = neo4j_id_to_graph_idx[rel.end_node.id]
        edge_list.append((start_node_idx, end_node_idx))
        edge_types.append(db_info['edge_type_to_int'][rel.type])

    with open(os.path.join(target_dir, str(dp_id)), 'wb') as f:
        dp_tuple = (edge_list, node_types, edge_types, features, label)
        pickle.dump(dp_tuple, f)


if __name__ == '__main__':
    db_name, dp_id, base_query, target_dir = sys.argv[1:]
    print('Doing {}'.format(dp_id))
    create_datapoint_from_database(db_name, base_query, target_dir, dp_id)
