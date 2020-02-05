from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss as XEntL, Parameter


class CrossEntropyLoss(nn.Module):
    def __init__(self, model, weight=None, **kwargs):
        super().__init__()
        if weight is not None:
            weight = torch.Tensor(weight)
        self.loss = XEntL(weight=weight)

    def forward(self, input, target):
        return self.loss(input, target)


class MSELoss(nn.Module):
    def forward(self, input, target):
        if target.dtype == torch.int64:
            target = F.one_hot(target, num_classes=2).to(torch.float32)
        return F.mse_loss(input, target, reduction='mean')


class FocalLoss(nn.Module):
    r"""
    https://arxiv.org/pdf/1708.02002.pdf
    """

    def __init__(self, model, gamma=2.0, weight: list = None, **kwargs):
        super().__init__()
        weight = torch.Tensor([1., 2.] if weight is None else weight)
        self.weight = Parameter(weight, requires_grad=False)
        self.gamma = gamma

    def forward(self, input, target):
        probs = F.softmax(input, dim=1)
        probs = torch.gather(probs, 1, target.unsqueeze(1))
        focal = (1 - probs) ** self.gamma
        nll = -1 * torch.log(probs)
        loss = focal * nll
        loss *= torch.index_select(self.weight, 0, target).unsqueeze(1)
        loss = torch.mean(loss)

        return loss


class DAEMLPLoss(nn.Module):
    """
    Denoising AutoEncoder (DAE) Loss.  Tries to reconstruct the input with an MLP, and penalizes the difference.
    Intended for use in pretraining TabMLP, but could be used in general.
    """

    def __init__(self, model, p_columns_to_shuffle, **kwargs):
        super().__init__()
        assert model.one_hot_embeddings
        model.layers = model.layers[:-3]  # So the model returns its final representation as the bottleneck

        layer_sizes = [m.weight.shape[0] for m in model.layers[::-1] if isinstance(m, torch.nn.Linear)]
        layers = []
        prev_layer_size = layer_sizes.pop(0)
        for layer_size in layer_sizes:
            layers.append(nn.Linear(prev_layer_size, layer_size))
            layers.append(model.get_act())
            layers.append(model.get_norm(layer_size))
            layers.append(nn.Dropout(model.p_dropout))
            prev_layer_size = layer_size
        layers.append(nn.Linear(prev_layer_size, model.init_feat_dim))
        self.layers = nn.Sequential(*layers)
        self.p_columns_to_shuffle = p_columns_to_shuffle
        self.mse = MSELoss()
        self.override_model_forward(model)

    def override_model_forward(self, model):
        orig_model_forward = model.forward
        p_columns_to_shuffle = self.p_columns_to_shuffle
        DAE_layers = self.layers

        def new_model_forward(input):
            # Copying the input and encoding it to match the output of the DAE
            status = model.training
            model.eval()
            with torch.no_grad():
                orig_cat_feats = [init(input[0][:, i]) for i, init in enumerate(model.cat_initializers.values())]
                if orig_cat_feats != []:
                    orig_cat_feats = [torch.cat(orig_cat_feats, dim=1)]
                orig_input = torch.cat(orig_cat_feats + ([input[1]] if not isinstance(input[1], list) else input[1]),
                                       dim=1)

                if isinstance(input[0], torch.Tensor):
                    n_cat = input[0].shape[1]
                    cat_cols_to_shuffle = np.random.choice(n_cat, int(p_columns_to_shuffle * n_cat), replace=False)
                    for col in cat_cols_to_shuffle:
                        input[0][:, col] = input[0][:, col][torch.randperm(input[0].shape[0])]
                if isinstance(input[1], torch.Tensor):
                    n_cont = input[1].shape[1]
                    cont_cols_to_shuffle = np.random.choice(n_cont, int(p_columns_to_shuffle * n_cont), replace=False)
                    for col in cont_cols_to_shuffle:
                        input[1][:, col] = input[1][:, col][torch.randperm(input[1].shape[0])]
            model.train(status)

            return DAE_layers(orig_model_forward(input)), orig_input

        model.forward = new_model_forward

    def forward(self, input, target):
        model_output, orig_input = input
        return self.mse(model_output, orig_input)


class MLMTabTransformerLoss(nn.Module):
    def __init__(self, model, p_mask, **kwargs):
        super().__init__()
        if model.n_cont_features != 0:
            print(
                f'WARNING: MLM only works with categorical inputs. and model has {model.n_cont_features} cont features')
        assert model.__class__.__name__ == 'TabTransformer'  # Hacky, but prevents circular import
        assert model.act_on_output == False
        assert model.readout == 'all_feat_embs'

        self.p_mask = p_mask
        # Init prediction head for every categorical variable
        self.cat_decoders = nn.ModuleDict()
        self.cards = []
        for col_name, card in model.cat_feat_origin_cards:
            self.cards.append(card)
            self.cat_decoders[col_name] = nn.Sequential(
                nn.Linear(model.hidden_dim, model.hidden_dim),
                model.get_act(),
                model.get_norm(model.hidden_dim),
                nn.Linear(model.hidden_dim, card)
            )
        self.xent = XEntL()
        self.override_model_forward(model)

    def override_model_forward(self, model):
        p_mask = self.p_mask

        def new_model_forward(input):
            cat_feats = input[0]
            orig_cat_feats = deepcopy(input[0].detach())

            feat_mask = torch.empty_like(cat_feats, dtype=float).uniform_() < p_mask
            replace_mask = (torch.empty_like(cat_feats, dtype=float).uniform_() < 0.8) & feat_mask
            random_mask = (torch.empty_like(cat_feats, dtype=float).uniform_() < 0.5) & feat_mask & ~replace_mask

            # Set the random words
            col_cardinalities = torch.LongTensor([i[1] for i in model.cat_feat_origin_cards]).to(cat_feats)
            col_cardinalities = col_cardinalities.unsqueeze(0).expand_as(cat_feats)
            unif = torch.rand(feat_mask.shape, device=col_cardinalities.device)
            random_feats = (unif * col_cardinalities).floor().to(torch.int64)
            cat_feats[random_mask] = random_feats[random_mask]

            feat_embs = model.init_input(input)

            # Mask out the features to be masked (we're implicitly assigning mask token to be the 0 vector)
            replace_mask = replace_mask.T.unsqueeze(2).expand_as(feat_embs)
            feat_embs = feat_embs * ~replace_mask

            # Set the entries of orig_cat_feats (the target) to be -100 if they weren't masked so XentL ignores them
            orig_cat_feats[~feat_mask] = -100

            return model.run_tfmr(feat_embs), orig_cat_feats

        model.forward = new_model_forward

    def forward(self, input, target):
        model_output, orig_cat_feats = input
        loss = torch.Tensor([0]).to(model_output)
        # The for loop isn't pretty, but pytorch xent seems to play badly with different cardinalities for each class
        for i, decoder in enumerate(self.cat_decoders.values()):
            l = decoder(model_output[i])
            loss = loss + self.xent(l, orig_cat_feats[:, i])
        return loss / len(self.cat_decoders)


class MLMTabBERTLoss(nn.Module):
    def __init__(self, model, p_mask, **kwargs):
        super().__init__()
        assert model.__class__.__name__ == 'TabBERT'  # Hacky, but prevents circular import
        assert model.act_on_output == False
        assert model.readout == 'all_feat_embs'

        self.p_mask = p_mask

        self.vocab_transform = nn.Linear(model.config.dim, model.config.dim)
        model.tfmr._init_weights(self.vocab_transform)
        self.vocab_act = model.get_act()
        self.vocab_norm = model.get_norm(model.config.dim)
        self.vocab_projector = nn.Linear(model.config.dim, model.config.vocab_size)
        model.tfmr._init_weights(self.vocab_projector)

        self.mlm_loss_fct = nn.CrossEntropyLoss()
        self.override_model_forward(model)

    def override_model_forward(self, model):
        p_mask = self.p_mask

        def new_model_forward(input):
            """This logic stolen with gratitude from https://github.com/huggingface/transformers"""
            inputs = model.init_input(input)

            labels = inputs.clone()
            # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
            probability_matrix = torch.full(labels.shape, p_mask, device=labels.device)
            special_tokens_mask = [
                model.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool, device=labels.device),
                                            value=0.0)
            if model.tokenizer._pad_token is not None:
                padding_mask = labels.eq(model.tokenizer.pad_token_id)
                probability_matrix.masked_fill_(padding_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100  # We only compute loss on masked tokens

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(
                torch.full(labels.shape, 0.8, device=labels.device)).bool() & masked_indices
            inputs[indices_replaced] = model.tokenizer.convert_tokens_to_ids(model.tokenizer.mask_token)

            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(
                torch.full(labels.shape, 0.5, device=labels.device)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(model.tokenizer), labels.shape, dtype=torch.long, device=labels.device)
            inputs[indices_random] = random_words[indices_random]

            # The rest of the time (10% of the time) we keep the masked input tokens unchanged
            return model.tfmr(inputs), labels

        model.forward = new_model_forward

    def forward(self, input, target):
        """This logic stolen with gratitude from https://github.com/huggingface/transformers"""
        tfmr_output, labels = input
        hidden_states = tfmr_output[0]  # (bs, seq_length, dim)
        prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_act(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_norm(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)

        mlm_loss = self.mlm_loss_fct(
            prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1)
        )

        return mlm_loss


class TabNetSparsityLoss(nn.Module):
    r"""
    Loss function that augments the cross entropy loss with an encouragement for the TabNet masks to be sparse
    """
    pass
    # Todo: finish this
