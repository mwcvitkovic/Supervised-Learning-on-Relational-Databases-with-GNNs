import collections
import glob
import json
import math
import os
from collections import Iterable
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from dgl import DGLGraph
from torch import nn
from torch.nn import Parameter
from tqdm import tqdm

sns.set(style="white", color_codes=True)

from utils import DummyWriter


def save_train_kwargs(writer, train_kwargs):
    if not isinstance(writer, DummyWriter):
        if writer.verbose:
            pprint(train_kwargs)
            print(f'Logging to {writer.log_dir}')
        if not isinstance(writer, DummyWriter):
            path = os.path.join(writer.log_dir, 'train_kwargs.json')
            with open(path, 'w') as f:
                json.dump(train_kwargs, f, indent=2)


def save_model_checkpoint(writer, epoch, model, optimizer, lr_sched, chkpt_name=None):
    if not isinstance(writer, DummyWriter):
        if chkpt_name is None:
            chkpt_name = epoch
        state_dict = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_sched': lr_sched}
        if isinstance(chkpt_name, str) and 'best' in chkpt_name:
            last_best_chkpt = glob.glob(os.path.join(writer.log_dir, f'model_checkpoint_{chkpt_name}*'))
            assert len(last_best_chkpt) < 2
            if last_best_chkpt:
                os.remove(last_best_chkpt[0])
            path = os.path.join(writer.log_dir, f'model_checkpoint_{chkpt_name}_{epoch}.pt')
        else:
            path = os.path.join(writer.log_dir, f'model_checkpoint_{chkpt_name}.pt')
        torch.save(state_dict, path)


def recursive_to(iterable, device):
    if isinstance(iterable, DGLGraph):
        iterable.to(device)
    if isinstance(iterable, torch.Tensor):
        iterable.data = iterable.data.to(device)
    elif isinstance(iterable, collections.abc.Mapping):
        for v in iterable.values():
            recursive_to(v, device)
    elif isinstance(iterable, (list, tuple)):
        for v in iterable:
            recursive_to(v, device)


def register_module_hooks(module_name, module, writer, dump_values=False):
    def log_forward_outputs(module, input, output):
        if not isinstance(output, torch.Tensor) and isinstance(output, Iterable):
            for i, o in enumerate(output):
                writer.add_histogram(f'Forward Outputs/{module_name}_{i}', o, writer.batches_done)
        else:
            writer.add_histogram(f'Forward Outputs/{module_name}', output, writer.batches_done)

    def log_final_representation_variance(module, input, output):
        out = pd.DataFrame(output.detach().cpu().numpy())
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.violinplot(x='activation', y='val', data=out.melt(var_name='activation', value_name='val'), ax=ax)
        writer.add_figure('Distribution of each activations in final representation', fig, writer.batches_done)
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.violinplot(x='batch', y='val', data=out.transpose().melt(var_name='batch', value_name='val'), ax=ax)
        writer.add_figure('Distribution of each batch element in final representation', fig, writer.batches_done)

    # def log_grad_inputs(module, grad_input, grad_output):
    #     for i, inp in enumerate(grad_input):
    #         if inp is not None:
    #             writer.add_histogram(f'Backward Inputs/{module_name}_grad_input_{i}', inp, writer.batches_done)

    for n, c in module.named_children():
        register_module_hooks('.'.join([module_name, n]), c, writer)
        if n == 'readout':
            c.register_forward_hook(log_final_representation_variance)

    module.register_forward_hook(log_forward_outputs)
    # module.register_backward_hook(log_grad_inputs)  # This more-or-less gets logged in log_param_values


def get_good_lr(model, optimizer, train_loader, init_value=1e-7, final_value=1.0, beta=0.0):
    """
    Find and return a good learning rate for this model with this optimizer.


    ***THIS MESSES UP THE MODEL AND OPTIMIZER - YOU NEED TO RESET THEM***


    """
    log_lrs, losses = test_lrs(model, optimizer, train_loader, init_value=init_value, final_value=final_value,
                               beta=beta)

    rmin_idx = np.where(np.array(losses) == min(losses))[0][-1]
    good_lr = (10 ** (log_lrs[rmin_idx])) / 100
    good_lr = max(min(good_lr, 1e-3), 1e-6)  # conservative safeguard

    from matplotlib import pyplot as plt
    plt.plot(log_lrs, [min(2.5, i) for i in losses], color='blue', label='losses')
    plt.plot(log_lrs[rmin_idx], losses[rmin_idx], color='red', marker='o', label='min lr')
    plt.plot(np.log10(good_lr), losses[rmin_idx], color='green', marker='o', label='selected lr')
    plt.legend()
    # plt.show()  # Debugging
    model.writer.add_figure('Test LogLR vs Loss', plt.gcf())

    return good_lr


def test_lrs(model, optimizer, train_loader, init_value=1e-9, final_value=10.0, beta=0.0):
    """
    Adapted from https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    Does a single epoch of training, sweeping through learning rates.
    Does not reset the model afterward.
    """
    num = len(train_loader) - 1
    mult = (final_value / init_value) ** (1 / num)
    lr = init_value
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    avg_loss = 0.0
    best_loss = 0.0
    batch_num = 0
    losses = []
    log_lrs = []
    model.train()
    for input, label in tqdm(train_loader):
        batch_num += 1
        # Get the loss for this mini-batch of inputs/outputs
        recursive_to((input, label), model.device)
        optimizer.zero_grad()
        output = model(input)
        loss = model.loss_fxn(output, label)
        # Compute the smoothed loss
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** batch_num)
        # if batch_num > 1 and smoothed_loss > 10 * best_loss:
        #     return log_lrs, losses
        # Record the best loss
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss
        # Store the values
        losses.append(smoothed_loss)
        log_lrs.append(np.log10(lr))
        # Do the SGD step
        loss.backward()
        optimizer.step()
        # Update the lr for the next step
        lr *= mult
        for pg in optimizer.param_groups:
            pg['lr'] = lr

    return log_lrs, losses


class TypeConditionalLinear(nn.Module):
    """
    Just like torch.nn.Linear, except each input gets multiplied by different weights depending its (integer) type
    """

    def __init__(self, in_features, out_features, n_types, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_types = n_types
        self.weight = Parameter(torch.Tensor(n_types, out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(n_types, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        for w in self.weight:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        if self.bias is not None:
            for i, b in enumerate(self.bias):
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[i])
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(b, -bound, bound)

    def forward(self, input, i_type):
        assert input.dtype == torch.float32 and i_type.dtype == torch.int64
        out = torch.empty(input.shape[0], self.out_features).fill_(np.nan).to(input)
        for t, w in enumerate(self.weight):
            idxs_this_type = (i_type == t).nonzero()[:, 0]
            in_this_type = input[idxs_this_type]
            out_this_type = in_this_type.matmul(w.t())
            if self.bias is not None:
                out_this_type += self.bias[t]
            out[idxs_this_type] = out_this_type

        return out

    # This is a correct but memory-intensive version of forward()
    # def forward(self, input, i_type):
    #     assert input.dtype == torch.float32 and i_type.dtype == torch.int64
    #     expanded_weights = self.weight.index_select(0, i_type)
    #     out = torch.bmm(expanded_weights, input.unsqueeze(2)).squeeze(2)
    #     if self.bias is not None:
    #         expanded_bias = self.bias.index_select(0, i_type)
    #         out = out + expanded_bias
    #     return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, n_types={}, bias={}'.format(
            self.in_features, self.out_features, self.n_types, self.bias is not None
        )
