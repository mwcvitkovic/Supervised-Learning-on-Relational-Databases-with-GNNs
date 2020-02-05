from dgl import BatchedDGLGraph

from models.GNN.GNNModelBase import GNNModelBase


class PoolMLP(GNNModelBase):
    """
    Model that ignores relational structure and just inits the nodes, pools them, and computes the output.
    Inspired by https://arxiv.org/pdf/1905.04682.pdf
    """

    def __init__(self, **kwargs):
        super().__init__(n_layers=0, **kwargs)

    def gnn_forward(self, g: BatchedDGLGraph):
        feats = g.ndata['h']
        readout = self.readout(g, feats)
        out = self.fcout(readout)
        return out
