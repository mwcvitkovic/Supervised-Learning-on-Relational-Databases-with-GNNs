import torch
import torch.nn.functional as F
from dgl import BatchedDGLGraph
from dgl.nn.pytorch.conv import GraphConv, RelGraphConv
from torch import nn

from models.GNN.GNNModelBase import GNNModelBase
from models.utils import TypeConditionalLinear


class GCN(GNNModelBase):
    """
    Graph Convolutional Network as described in https://arxiv.org/pdf/1609.02907.pdf
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(nn.ModuleDict({
                'gc': GraphConv(in_feats=self.hidden_dim,
                                out_feats=self.hidden_dim,
                                norm=True,
                                bias=True,
                                activation=self.get_act()),
                'norm': self.get_norm(self.hidden_dim),
                'do': nn.Dropout(self.p_dropout)
            }))

    def gnn_forward(self, g: BatchedDGLGraph):
        feats = g.ndata['h']
        for block in self.layers:
            feats = block['gc'](g, feats)
            feats = block['norm'](feats)
            feats = block['do'](feats)
        readout = self.readout(g, feats)
        out = self.fcout(readout)
        return out


class RelationalGCN(GNNModelBase):
    """
    Relational Graph Convolutional Network as described in https://arxiv.org/abs/1703.06103
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layers = nn.ModuleList()
        self.n_relations = 2 * len(
            self.db_info['edge_type_to_int']) - 1  # there are negative edge types for the reverse edges
        for _ in range(self.n_layers):
            self.layers.append(nn.ModuleDict({'rgc': RelGraphConv(in_feat=self.hidden_dim,
                                                                  out_feat=self.hidden_dim,
                                                                  num_rels=self.n_relations,
                                                                  regularizer='bdd',
                                                                  num_bases=8,
                                                                  bias=True,
                                                                  dropout=self.p_dropout,
                                                                  activation=self.get_act(),
                                                                  self_loop=False,  # It's already in the data
                                                                  ),
                                              'norm': self.get_norm(self.hidden_dim)
                                              }))

    def gnn_forward(self, g: BatchedDGLGraph):
        feats = g.ndata['h']
        etypes = g.edata['edge_types'] + self.n_relations // 2
        for block in self.layers:
            feats = block['norm'](feats)
            feats = block['rgc'](graph=g, x=feats, etypes=etypes)
        readout = self.readout(g, feats)
        out = self.fcout(readout)
        return out


class ERGCN(GNNModelBase):
    """
    GCN using different linear mappings for each node and edge type
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layers = nn.ModuleList()
        self.n_node_types = len(self.db_info['node_type_to_int'])
        self.n_relations = 2 * len(
            self.db_info['edge_type_to_int']) - 1  # there are negative edge types for the reverse edges
        for _ in range(self.n_layers):
            self.layers.append(nn.ModuleDict({'ergc': ERGCNConv(in_feat=self.hidden_dim,
                                                                out_feat=self.hidden_dim,
                                                                n_node_types=self.n_node_types,
                                                                n_rels=self.n_relations,
                                                                p_dropout=self.p_dropout,
                                                                activation=self.get_act()
                                                                ),
                                              'norm': self.get_norm(self.hidden_dim)
                                              }))

    def gnn_forward(self, g: BatchedDGLGraph):
        feats = g.ndata['h']
        ntypes = g.ndata['node_types']
        etypes = g.edata['edge_types'] + self.n_relations // 2
        for block in self.layers:
            feats = block['norm'](feats)
            feats = block['ergc'](graph=g, feats=feats, ntypes=ntypes, etypes=etypes)
        readout = self.readout(g, feats)
        out = self.fcout(readout)
        return out


class ERGCNConv(nn.Module):
    def __init__(self, in_feat, out_feat, n_node_types, n_rels, p_dropout, activation):
        super().__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.n_node_types = n_node_types
        self.n_rels = n_rels
        self.p_dropout = p_dropout
        self.activation = activation
        self.fc_node = TypeConditionalLinear(in_feat, out_feat, n_node_types)
        self.fc_edge = TypeConditionalLinear(in_feat, out_feat, n_rels)

    def message_func(self, edges):
        msg = edges.src['h']
        etypes = edges.data['type']
        msg = self.fc_edge(msg, etypes)
        return {'msg': msg, 'etype': etypes}

    def update_func(self, nodes):
        """Aggregates the messages, but doesn't do the complete hidden state update"""
        in_msg = nodes.mailbox['msg']
        # Normalize by edge type
        in_etype = nodes.mailbox['etype'].detach()[0]
        counts = F.one_hot(in_etype).sum(axis=0, keepdim=True)
        counts = torch.index_select(counts, 1, in_etype).unsqueeze(2)
        in_msg /= counts
        in_msg = torch.sum(in_msg, axis=1)
        return {'msg': in_msg}

    def forward(self, graph, feats, ntypes, etypes):
        # Pass and aggregate messages
        graph = graph.local_var()
        graph.ndata['h'] = feats
        graph.edata['type'] = etypes
        graph.update_all(self.message_func, self.update_func)
        # Update hidden state
        feats = graph.ndata['h']
        msg = graph.ndata['msg']
        feats = self.fc_node(feats, ntypes)
        feats += msg
        feats = self.activation(feats)
        F.dropout(feats, p=self.p_dropout, training=self.training)
        return feats
