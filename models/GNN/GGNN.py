from dgl import BatchedDGLGraph
from dgl.nn.pytorch.conv import GatedGraphConv
from torch import nn

from models.GNN.GNNModelBase import GNNModelBase


class RelationalGGNN(GNNModelBase):
    """
    Gated Graph Convolution as described in https://arxiv.org/pdf/1511.05493.pdf
    """

    # todo: see if this is still too expensive to run

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layers = nn.ModuleList()
        self.n_relations = 2 * len(
            self.db_info['edge_type_to_int']) - 1  # there are negative edge types for the reverse edges
        for _ in range(self.n_layers):
            self.layers.append(nn.ModuleDict({'norm': self.get_norm(self.hidden_dim),
                                              'do': nn.Dropout(self.p_dropout),
                                              'ggc': GatedGraphConv(in_feats=self.hidden_dim,
                                                                    out_feats=self.hidden_dim,
                                                                    n_steps=3,
                                                                    n_etypes=self.n_relations,
                                                                    bias=True,
                                                                    )}))

    def gnn_forward(self, g: BatchedDGLGraph):
        feats = g.ndata['h']
        etypes = g.edata['edge_types'] + self.n_relations // 2
        for block in self.layers:
            feats = block['norm'](feats)
            feats = block['do'](feats)
            feats = block['ggc'](graph=g, feat=feats, etypes=etypes)
        readout = self.readout(g, feats)
        out = self.fcout(readout)
        return out
