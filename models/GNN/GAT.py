import dgl.function as fn
import torch
from dgl import BatchedDGLGraph
from dgl.nn.pytorch import edge_softmax
from dgl.nn.pytorch.conv import GATConv
from torch import nn

from models.GNN.GNNModelBase import GNNModelBase
from models.utils import TypeConditionalLinear


class GAT(GNNModelBase):
    """
    Graph Attention Network as described in https://arxiv.org/pdf/1710.10903.pdf
    """

    def __init__(self, n_heads, residual, **kwargs):
        super().__init__(**kwargs)
        assert self.hidden_dim % n_heads == 0, 'hidden_dim needs to be divisible by n_heads for shapes to align'
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(GATConv(in_feats=self.hidden_dim,
                                       out_feats=self.hidden_dim // n_heads,
                                       num_heads=n_heads,
                                       feat_drop=self.p_dropout,
                                       attn_drop=self.p_dropout,
                                       residual=residual,
                                       activation=self.get_act()))

    def gnn_forward(self, g: BatchedDGLGraph):
        feats = g.ndata['h']
        for layer in self.layers:
            feats = layer(graph=g, feat=feats)
            feats = feats.reshape(feats.shape[0], -1)
        readout = self.readout(g, feats)
        out = self.fcout(readout)
        return out


class RelationalGAT(GNNModelBase):
    """
    Relational version of Graph Attention Network
    """

    def __init__(self, n_heads, residual, **kwargs):
        super().__init__(**kwargs)
        assert self.hidden_dim % n_heads == 0, 'hidden_dim needs to be divisible by n_heads for shapes to align'
        self.n_relations = 2 * len(
            self.db_info['edge_type_to_int']) - 1  # there are negative edge types for the reverse edges
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(RelationalGATConv(in_feats=self.hidden_dim,
                                                 out_feats=self.hidden_dim // n_heads,
                                                 num_heads=n_heads,
                                                 feat_drop=self.p_dropout,
                                                 attn_drop=self.p_dropout,
                                                 residual=residual,
                                                 activation=self.get_act(),
                                                 num_rels=self.n_relations))

    def gnn_forward(self, g: BatchedDGLGraph):
        feats = g.ndata['h']
        etypes = g.edata['edge_types'] + self.n_relations // 2
        for layer in self.layers:
            feats = layer(graph=g, feat=feats, etypes=etypes)
            feats = feats.reshape(feats.shape[0], -1)
        readout = self.readout(g, feats)
        out = self.fcout(readout)
        return out


class RelationalGATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.0,
                 attn_drop=0.0,
                 residual=False,
                 activation=None,
                 num_rels=0):
        super().__init__()
        self.num_rels = num_rels
        self.num_heads = num_heads
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=True)
        self.r_attn_fcs = nn.ModuleList(
            [TypeConditionalLinear(in_feats, out_feats, self.num_rels, bias=True) for _ in range(num_heads)])
        self.l_attn_fcs = nn.ModuleList(
            [TypeConditionalLinear(in_feats, out_feats, self.num_rels, bias=True) for _ in range(num_heads)])
        self.out_attn_fcs = nn.ModuleList(
            [TypeConditionalLinear(out_feats * 2, 1, self.num_rels, bias=True) for _ in range(num_heads)])
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        if residual:
            if in_feats != out_feats:
                self.res_fc = nn.Linear(in_feats, num_heads * out_feats, bias=True)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)
        self.activation = activation

    def message_func(self, edges):
        h_src = edges.src['h']
        h_dst = edges.dst['h']
        etypes = edges.data['type']
        l_msg = [attn_fc(h_src, etypes) for attn_fc in self.l_attn_fcs]
        r_msg = [attn_fc(h_dst, etypes) for attn_fc in self.r_attn_fcs]
        msg = [torch.cat([l, r], dim=1) for l, r in zip(l_msg, r_msg)]
        msg = [attn_fc(m, etypes) for m, attn_fc in zip(msg, self.out_attn_fcs)]
        msg = torch.stack(msg, axis=1)
        return {'msg': msg}

    def forward(self, graph, feat, etypes):
        graph = graph.local_var()
        h = self.feat_drop(feat)
        feat = self.fc(h).view(-1, self.num_heads, self.out_feats)
        graph.ndata.update({'ft': feat})
        graph.edata['type'] = etypes
        graph.apply_edges(self.message_func)
        e = self.activation(graph.edata.pop('msg'))
        graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
        # message passing
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        rst = graph.ndata['ft']
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h).view(-1, self.num_heads, self.out_feats)
            rst = rst + resval
        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst


class ERGAT(RelationalGAT):
    """
    GAT using different linear mappings for each node and edge type

    todo: compare to this relational GAT model: https://openreview.net/pdf?id=Bklzkh0qFm
    """

    def __init__(self, n_heads, residual, **kwargs):
        super().__init__(n_heads, residual, **kwargs)
        self.n_node_types = len(self.db_info['node_type_to_int'])
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(ERGATConv(in_feats=self.hidden_dim,
                                         out_feats=self.hidden_dim // n_heads,
                                         num_heads=n_heads,
                                         feat_drop=self.p_dropout,
                                         attn_drop=self.p_dropout,
                                         residual=residual,
                                         activation=self.get_act(),
                                         n_node_types=self.n_node_types,
                                         num_rels=self.n_relations))

    def gnn_forward(self, g: BatchedDGLGraph):
        feats = g.ndata['h']
        ntypes = g.ndata['node_types']
        etypes = g.edata['edge_types'] + self.n_relations // 2
        for layer in self.layers:
            feats = layer(graph=g, feat=feats, ntypes=ntypes, etypes=etypes)
            feats = feats.reshape(feats.shape[0], -1)
        readout = self.readout(g, feats)
        out = self.fcout(readout)
        return out


class ERGATConv(RelationalGATConv):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.0,
                 attn_drop=0.0,
                 residual=False,
                 activation=None,
                 n_node_types=0,
                 num_rels=0):
        super().__init__(in_feats,
                         out_feats,
                         num_heads,
                         feat_drop,
                         attn_drop,
                         residual,
                         activation,
                         num_rels)
        self.fc = TypeConditionalLinear(in_feats, out_feats * num_heads, n_node_types, bias=True)
        if residual:
            if in_feats != out_feats:
                self.res_fc = TypeConditionalLinear(in_feats, num_heads * out_feats, n_node_types, bias=True)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)

    def forward(self, graph, feat, ntypes, etypes):
        graph = graph.local_var()
        h = self.feat_drop(feat)
        feat = self.fc(h, ntypes).view(-1, self.num_heads, self.out_feats)
        graph.ndata.update({'ft': feat})
        graph.edata['type'] = etypes
        graph.apply_edges(self.message_func)
        e = self.activation(graph.edata.pop('msg'))
        graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
        # message passing
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        rst = graph.ndata['ft']
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h, ntypes).view(-1, self.num_heads, self.out_feats)
            rst = rst + resval
        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst
