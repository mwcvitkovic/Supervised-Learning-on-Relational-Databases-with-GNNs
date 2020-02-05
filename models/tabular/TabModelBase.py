from torch import nn

from data.data_encoders import EmbeddingInitializer
from data.utils import get_ds_info
from models import activations, losses


class TabModelBase(nn.Module):
    """
    Base class for all tabular models
    """

    def __init__(self, writer, dataset_name, n_cont_features, cat_feat_origin_cards, max_emb_dim,
                 activation_class_name, activation_class_kwargs, norm_class_name, norm_class_kwargs, p_dropout,
                 one_hot_embeddings, drop_whole_embeddings, loss_class_name=None, loss_class_kwargs=None,
                 n_out=None):
        super().__init__()
        self.writer = writer
        if dataset_name is not None:
            assert n_out is None
            self.ds_info = get_ds_info(dataset_name)
            task = self.ds_info['processed']['task']
            if task == 'binary classification':
                self.n_out = 2
            elif task == 'multiclass classification':
                raise NotImplementedError  # todo
            elif task == 'regression':
                self.n_out = 1
            self.act_on_output = False
        else:
            assert n_out is not None
            self.n_out = n_out
            self.act_on_output = True
        self.n_cont_features = n_cont_features
        self.cat_feat_origin_cards = cat_feat_origin_cards

        self.p_dropout = p_dropout
        self.drop_whole_embeddings = drop_whole_embeddings
        self.one_hot_embeddings = one_hot_embeddings
        self.act_class = activations.__dict__[activation_class_name]
        self.act_class_kwargs = activation_class_kwargs
        self.norm_class = nn.__dict__[norm_class_name]
        self.norm_class_kwargs = norm_class_kwargs
        self.loss_class_name = loss_class_name
        self.loss_class_kwargs = loss_class_kwargs
        self.cat_initializers = nn.ModuleDict()
        if isinstance(self.cat_feat_origin_cards, list):
            for col_name, card in self.cat_feat_origin_cards:
                self.cat_initializers[col_name] = EmbeddingInitializer(card, max_emb_dim, p_dropout,
                                                                       drop_whole_embeddings=drop_whole_embeddings,
                                                                       one_hot=one_hot_embeddings)
            self.init_feat_dim = sum(i.emb_dim for i in self.cat_initializers.values()) + self.n_cont_features

    def init_loss_fxn(self):
        if self.loss_class_name is not None and self.loss_class_kwargs is not None:
            self.loss_fxn = losses.__dict__[self.loss_class_name](self, **self.loss_class_kwargs)

    def get_act(self):
        return self.act_class(**self.act_class_kwargs)

    def get_norm(self, num_feats):
        return self.norm_class(num_feats, **self.norm_class_kwargs)

    def forward(self, input):
        """
        Returns logits for output classes
        """
        raise NotImplementedError

    def pred_from_output(self, output):
        """
        Returns the model's prediction of the class of the input given the output of self.forward
        """
        return output.max(dim=1, keepdim=True)[1]
