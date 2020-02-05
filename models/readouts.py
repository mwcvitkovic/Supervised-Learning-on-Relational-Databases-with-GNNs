import torch.nn as nn
from dgl.nn.pytorch import AvgPooling as AP, SortPooling as SP, GlobalAttentionPooling as GAP, Set2Set as S2S, \
    SetTransformerDecoder as STD

from models import activations


class AvgPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.ap = AP()

    def forward(self, graph, feat):
        return self.ap(graph, feat)


class SortPooling(nn.Module):
    def __init__(self, hidden_dim, k):
        super().__init__()
        self.sp = SP(k=k)
        self.fc = nn.Linear(hidden_dim * k, hidden_dim)

    def forward(self, graph, feat):
        feat = self.sp(graph, feat)
        return self.fc(feat)


class GlobalAttentionPooling(nn.Module):
    def __init__(self, hidden_dim, n_layers, act_name):
        super().__init__()
        act_class = activations.__dict__[act_name]
        gate_nn_layers = [l for _ in range(n_layers) for l in (nn.Linear(hidden_dim, hidden_dim), act_class())]
        gate_nn_layers.append(nn.Linear(hidden_dim, 1))
        gate_nn = nn.Sequential(*gate_nn_layers)
        feat_nn = nn.Sequential(
            *[l for _ in range(n_layers) for l in (nn.Linear(hidden_dim, hidden_dim), act_class())])
        self.gap = GAP(gate_nn=gate_nn, feat_nn=feat_nn)

    def forward(self, graph, feat):
        return self.gap(graph, feat)


class Set2Set(nn.Module):
    def __init__(self, hidden_dim, n_iters, n_layers):
        super().__init__()
        self.s2s = S2S(input_dim=hidden_dim, n_iters=n_iters, n_layers=n_layers)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, graph, feat):
        feat = self.s2s(graph, feat)
        return self.fc(feat)


class SetTransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, p_dropout, num_heads, n_layers, k):
        super().__init__()
        self.std = STD(d_model=hidden_dim,
                       num_heads=num_heads,
                       d_head=hidden_dim,
                       d_ff=hidden_dim,
                       n_layers=n_layers,
                       k=k,
                       dropouth=p_dropout,
                       dropouta=p_dropout)
        self.fc = nn.Linear(hidden_dim * k, hidden_dim)

    def forward(self, graph, feat):
        feat = self.std(graph, feat)
        return self.fc(feat)
