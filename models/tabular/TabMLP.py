import torch
from torch import nn

from models.tabular.TabModelBase import TabModelBase


class TabMLP(TabModelBase):
    """
    Straightforward MLP model for tabular data, loosely based on github.com/fastai/fastai/blob/master/fastai/tabular

    layer_sizes can contain ints, indicating the # of hidden units, or floats, indicating multiples of init_feat_dim
    """

    def __init__(self, layer_sizes, **kwargs):
        super().__init__(**kwargs)
        self.cont_norm = self.get_norm(self.n_cont_features)
        if all(isinstance(s, float) for s in layer_sizes):
            layer_sizes = [int(self.init_feat_dim * s) for s in layer_sizes]
        assert all(isinstance(s, int) for s in layer_sizes)
        self.layer_sizes = layer_sizes
        layers = []
        prev_layer_size = self.init_feat_dim
        for layer_size in self.layer_sizes:
            layers.append(nn.Linear(prev_layer_size, layer_size))
            layers.append(self.get_act())
            layers.append(self.get_norm(layer_size))
            layers.append(nn.Dropout(self.p_dropout))
            prev_layer_size = layer_size
        layers.append(nn.Linear(prev_layer_size, self.n_out))
        self.layers = nn.Sequential(*layers)
        self.init_loss_fxn()

    def forward(self, input):
        """
        Returns logits for output classes
        """
        # t = time.perf_counter()
        cat_feats, cont_feats = input
        cat_feats = [init(cat_feats[:, i]) for i, init in enumerate(self.cat_initializers.values())]
        if cat_feats != []:
            cat_feats = torch.cat(cat_feats, dim=1)
        # if self.training:
        #     self.writer.add_scalar('CodeProfiling/Model/TabMLPInitCatFeatures', time.perf_counter() - t,
        #                            self.writer.batches_done)
        # t = time.perf_counter()
        if isinstance(cont_feats, torch.Tensor):
            cont_feats = self.cont_norm(cont_feats)

        if isinstance(cat_feats, torch.Tensor) and isinstance(cont_feats, torch.Tensor):
            feats = torch.cat((cat_feats, cont_feats), dim=1)
        elif isinstance(cat_feats, torch.Tensor):
            feats = cat_feats
        else:
            feats = cont_feats
        out = self.layers(feats)
        if self.act_on_output:
            out = self.get_act()(out)
        # if self.training:
        #     self.writer.add_scalar('CodeProfiling/Model/TabMLPLayers', time.perf_counter() - t,
        #                            self.writer.batches_done)
        return out


class TabLogReg(TabMLP):
    def __init__(self, **kwargs):
        layer_sizes = []
        super().__init__(layer_sizes, **kwargs)
