import time

import numpy as np
import torch
from dgl import BatchedDGLGraph
from torch import nn

import models.tabular as tab_models
from data.utils import get_db_info
from models import losses, activations
from models import readouts


class GNNModelBase(nn.Module):
    """
    Base class for all GNN models
    """

    def __init__(self, writer, dataset_name, feature_encoders, hidden_dim, init_model_class_name, init_model_kwargs,
                 n_layers, activation_class_name, activation_class_kwargs, norm_class_name, norm_class_kwargs,
                 loss_class_kwargs, loss_class_name, p_dropout, readout_class_name, readout_kwargs, fcout_layer_sizes):
        super(GNNModelBase, self).__init__()
        self.writer = writer
        self.db_info = get_db_info(dataset_name)
        self.n_out = self.db_info['task']['n_classes']
        self.feature_encoders = feature_encoders
        self.init_model_class = tab_models.__dict__[init_model_class_name]
        self.init_model_kwargs = init_model_kwargs
        self.hidden_dim = hidden_dim
        self.p_dropout = p_dropout
        self.n_layers = n_layers
        if loss_class_kwargs.get('weight', None):
            loss_class_kwargs['weight'] = torch.Tensor(loss_class_kwargs['weight'])
        self.act_class = activations.__dict__[activation_class_name]
        self.act_class_kwargs = activation_class_kwargs
        self.norm_class = nn.__dict__[norm_class_name]
        self.norm_class_kwargs = norm_class_kwargs
        self.loss_fxn = losses.__dict__[loss_class_name](self, **loss_class_kwargs)

        # Create self.initializers for use in self.init_batch
        self.node_initializers = nn.ModuleDict()
        self.node_init_info = {}
        for node_type, features in self.db_info['node_types_and_features'].items():
            cat_feat_origin_cards = []
            cont_feat_origin = []
            for feature_name, feature_info in features.items():
                if '{}.{}'.format(node_type, feature_name) != self.db_info['label_feature']:
                    enc = self.feature_encoders[node_type][feature_name]
                    cat_feat_origin_cards += [(f'{feature_name}_{i}', card) for i, card in enumerate(enc.cat_cards)]
                    cont_feat_origin += [feature_name] * enc.cont_dim
            self.node_init_info[node_type] = {
                'cat_feat_origin_cards': cat_feat_origin_cards,
                'cont_feat_origin': cont_feat_origin,
            }
            self.node_initializers[node_type] = self.init_model_class(writer=writer,
                                                                      dataset_name=None,
                                                                      n_cont_features=len(cont_feat_origin),
                                                                      cat_feat_origin_cards=cat_feat_origin_cards,
                                                                      n_out=hidden_dim,
                                                                      **self.init_model_kwargs)

        # Create readout function
        self.readout = readouts.__dict__[readout_class_name](hidden_dim=hidden_dim, **readout_kwargs)

        # Create MLP "fcout" to produce output of model from output of readout
        if all(isinstance(s, float) for s in fcout_layer_sizes):
            fcout_layer_sizes = [int(self.hidden_dim * s) for s in fcout_layer_sizes]
        assert all(isinstance(s, int) for s in fcout_layer_sizes)
        self.layer_sizes = fcout_layer_sizes
        fcout_layers = []
        prev_layer_size = self.hidden_dim
        for layer_size in self.layer_sizes:
            fcout_layers.append(nn.Linear(prev_layer_size, layer_size))
            fcout_layers.append(self.get_act())
            fcout_layers.append(self.get_norm(layer_size))
            fcout_layers.append(nn.Dropout(self.p_dropout))
            prev_layer_size = layer_size
        fcout_layers.append(nn.Linear(prev_layer_size, self.n_out))
        self.fcout = nn.Sequential(*fcout_layers)

    def get_act(self):
        return self.act_class(**self.act_class_kwargs)

    def get_norm(self, num_feats):
        return self.norm_class(num_feats, **self.norm_class_kwargs)

    def init_batch(self, bdgl: BatchedDGLGraph, b_features):
        """
        Uses the tabular models in self.node_initializers to encode the raw database features (datetimes, text, etc.) of
        each table, such that all nodes in bdgl have the same hidden state size.

        (Note: some of the encoding actually happens during data loading, for efficiency.  See the __init__ method of
        DatabaseDataset and the get_DGL_collator function.)

        This method is run before self.gnn_forward
        """
        b_node_types = bdgl.ndata['node_types']
        bdgl.ndata['h'] = torch.empty(bdgl.number_of_nodes(), self.hidden_dim, device=b_node_types.device)
        bdgl.ndata['h'][:] = np.nan
        for node_type, collated_features in b_features.items():
            # Compute the initial features for this node type...
            node_features = self.node_initializers[node_type](collated_features)

            # Scatter these features to the appropriate entries in bdgl.ndata
            node_type_int = self.db_info['node_type_to_int'][node_type]
            idxs_this_node_type = (b_node_types == node_type_int).nonzero()[:, 0]
            bdgl.nodes[idxs_this_node_type].data['h'] = node_features

        return bdgl

    def forward(self, input):
        """
        Returns logits for output classes
        """
        bdgl, features = input
        # t = time.perf_counter()
        g = self.init_batch(bdgl, features)
        # if self.training:
        #   self.writer.add_scalar('CodeProfiling/Model/init_batch', time.perf_counter() - t, self.writer.batches_done)
        #
        t = time.perf_counter()
        out = self.gnn_forward(g)
        # if self.training:
        #   self.writer.add_scalar('CodeProfiling/Model/gnn_forward', time.perf_counter() - t, self.writer.batches_done)
        return out

    def gnn_forward(self, g: BatchedDGLGraph):
        """
        Runs the GNN component of the model and returns logits for output classes.

        :param g: BatchedDGLGraph with g.ndata[h] initialized to a (n_nodes x hidden_dim) tensor by self.init_batch
        """
        raise NotImplementedError

    def pred_from_output(self, output):
        """
        Returns the model's prediction of the class of the input given the output of self.forward
        """
        return output.max(dim=1, keepdim=True)[1]
