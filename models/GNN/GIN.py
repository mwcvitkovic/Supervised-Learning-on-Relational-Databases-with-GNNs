from dgl import BatchedDGLGraph
from dgl.nn.pytorch.conv import GINConv
from torch import nn

from models.GNN.GNNModelBase import GNNModelBase
from models.utils import TypeConditionalLinear


class GIN(GNNModelBase):
    """
    Graph Isomorphism Network as described in https://arxiv.org/pdf/1810.00826.pdf
    """

    def __init__(self, n_apply_func_layers, aggregator_type, init_eps, learn_eps, **kwargs):
        super().__init__(**kwargs)
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            apply_func_layers = sum(
                [[nn.Linear(self.hidden_dim, self.hidden_dim),
                  self.get_act(),
                  self.get_norm(self.hidden_dim),
                  nn.Dropout(self.p_dropout)] for _ in
                 range(n_apply_func_layers)],
                [])
            apply_func = nn.Sequential(*apply_func_layers)
            self.layers.append(GINConv(apply_func=apply_func,
                                       aggregator_type=aggregator_type,
                                       init_eps=init_eps,
                                       learn_eps=learn_eps))

    def gnn_forward(self, g: BatchedDGLGraph):
        feats = g.ndata['h']
        for layer in self.layers:
            feats = layer(graph=g, feat=feats)
        readout = self.readout(g, feats)
        out = self.fcout(readout)
        return out


class RelationalGIN(GNNModelBase):
    """
    Version of GIN that passes edge-type-conditional messages
    """

    def __init__(self, n_apply_func_layers, aggregator_type, init_eps, learn_eps, **kwargs):
        super().__init__(**kwargs)
        self.n_relations = 2 * len(
            self.db_info['edge_type_to_int']) - 1  # there are negative edge types for the reverse edges
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            apply_func_layers = sum(
                [[nn.Linear(self.hidden_dim, self.hidden_dim),
                  self.get_act(),
                  self.get_norm(self.hidden_dim),
                  nn.Dropout(self.p_dropout)] for _ in
                 range(n_apply_func_layers)],
                [])
            apply_func = nn.Sequential(*apply_func_layers)
            self.layers.append(RelationalGINConv(apply_func=apply_func,
                                                 activation=self.get_act(),
                                                 aggregator_type=aggregator_type,
                                                 hidden_dim=self.hidden_dim,
                                                 init_eps=init_eps,
                                                 learn_eps=learn_eps,
                                                 num_rels=self.n_relations))

    def gnn_forward(self, g: BatchedDGLGraph):
        feats = g.ndata['h']
        etypes = g.edata['edge_types'] + self.n_relations // 2
        for layer in self.layers:
            feats = layer(graph=g, feat=feats, etypes=etypes)
        readout = self.readout(g, feats)
        out = self.fcout(readout)
        return out


class RelationalGINConv(GINConv):
    def __init__(self, apply_func, activation, aggregator_type, hidden_dim, init_eps=0, learn_eps=False, num_rels=0):
        super().__init__(apply_func, aggregator_type, init_eps, learn_eps)
        self.num_rels = num_rels
        self.act = activation
        self.edge_message_layer = TypeConditionalLinear(hidden_dim, hidden_dim, num_rels)

    def message_func(self, edges):
        msg = edges.src['h']
        msg = self.edge_message_layer(msg, edges.data['type'])
        msg = self.act(msg)
        return {'msg': msg}

    def forward(self, graph, feat, etypes):
        graph = graph.local_var()
        graph.ndata['h'] = feat
        graph.edata['type'] = etypes
        graph.update_all(self.message_func, self._reducer('msg', 'neigh'))
        rst = (1 + self.eps) * feat + graph.ndata['neigh']
        if self.apply_func is not None:
            rst = self.apply_func(rst)
        return rst


class ERGIN(RelationalGIN):
    """
    GIN using different linear mappings for each node and edge type
    """

    def __init__(self, n_apply_func_layers, aggregator_type, init_eps, learn_eps, **kwargs):
        super().__init__(n_apply_func_layers, aggregator_type, init_eps, learn_eps, **kwargs)
        self.n_node_types = len(self.db_info['node_type_to_int'])
        self.act = self.get_act()
        self.layers = nn.ModuleList()
        self.apply_func_blocks = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(RelationalGINConv(apply_func=None,
                                                 activation=self.get_act(),
                                                 aggregator_type=aggregator_type,
                                                 hidden_dim=self.hidden_dim,
                                                 init_eps=init_eps,
                                                 learn_eps=learn_eps,
                                                 num_rels=self.n_relations))
            self.apply_func_blocks.append(
                nn.ModuleList([nn.ModuleDict({'tcl': TypeConditionalLinear(self.hidden_dim,
                                                                           self.hidden_dim,
                                                                           self.n_node_types),
                                              'act': self.get_act(),
                                              'norm': self.get_norm(self.hidden_dim),
                                              'do': nn.Dropout(self.p_dropout)
                                              })
                               for _ in range(n_apply_func_layers)])
            )

    def gnn_forward(self, g: BatchedDGLGraph):
        feats = g.ndata['h']
        ntypes = g.ndata['node_types']
        etypes = g.edata['edge_types'] + self.n_relations // 2
        for layer, apply_func_blocks in zip(self.layers, self.apply_func_blocks):
            feats = layer(graph=g, feat=feats, etypes=etypes)
            for block in apply_func_blocks:
                feats = block['tcl'](feats, ntypes)
                feats = block['act'](feats)
                feats = block['norm'](feats)
                feats = block['do'](feats)
        readout = self.readout(g, feats)
        out = self.fcout(readout)
        return out
