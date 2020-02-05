import base64
import json
import os
import pickle
import pprint
import sys

import numpy as np
import torch
import torch.optim as opt

import models
from data.TabularDataset import TabularDataset
from models.tabular.TabModelBase import TabModelBase
from models.utils import register_module_hooks
from start_training import train_epoch
from utils import get_train_val_test_datasets, get_dataloader, setup_writer, model_to_device, \
    get_optim_with_correct_wd


def debug_epoch(
        model_logdir,
        checkpoint_id,
        device,
        num_workers):
    log_dir = os.path.join('debug', model_logdir, f'model_checkpoint_{checkpoint_id}')
    writer = setup_writer(log_dir, debug_network=True)
    print(f'Logging to {writer.log_dir}')

    with open(os.path.join('runs', model_logdir, 'train_kwargs.json')) as f:
        train_kwargs = json.load(f)
    train_data, _, _ = get_train_val_test_datasets(dataset_name=train_kwargs['dataset_name'],
                                                   train_test_split=train_kwargs['train_test_split'],
                                                   encoders=train_kwargs['encoders'],
                                                   train_fraction_to_use=train_kwargs['train_fraction_to_use'])
    train_loader = get_dataloader(dataset=train_data,
                                  batch_size=train_kwargs['batch_size'],
                                  sampler_class_name=train_kwargs['sampler_class_name'],
                                  sampler_class_kwargs=train_kwargs['sampler_class_kwargs'],
                                  num_workers=num_workers,
                                  max_nodes_per_graph=train_kwargs['max_nodes_per_graph'])

    checkpoint_path = os.path.join('runs', model_logdir, f'model_checkpoint_{checkpoint_id}.pt')
    state_dict = torch.load(checkpoint_path)

    model_class = models.__dict__[train_kwargs['model_class_name']]
    if isinstance(train_data, TabularDataset):
        assert issubclass(model_class, TabModelBase)
        train_kwargs['model_kwargs'].update(
            n_cont_features=train_data.n_cont_features,
            cat_feat_origin_cards=train_data.cat_feat_origin_cards
        )
    model = model_class(writer=writer,
                        dataset_name=train_kwargs['dataset_name'],
                        encoders=train_kwargs['encoders'],
                        **train_kwargs['model_kwargs'])
    model.load_state_dict(state_dict['model'])
    model_to_device(model, device)
    register_module_hooks('model', model, writer)

    optimizer = get_optim_with_correct_wd(train_kwargs['optimizer_class_name'], model,
                                          train_kwargs['optimizer_kwargs'],
                                          train_kwargs['wd_bias'],
                                          train_kwargs['wd_embed'],
                                          train_kwargs['wd_bn'])
    optimizer.load_state_dict(state_dict['optimizer'])
    scheduler = opt.lr_scheduler.__dict__[train_kwargs['lr_scheduler_class_name']](optimizer, **train_kwargs[
        'lr_scheduler_kwargs'])

    # Run a single epoch and save the debugging information
    pprint.pprint(train_kwargs)
    writer.add_text('train_kwargs', pprint.pformat(train_kwargs).replace('\n', '\t\n'))
    epoch = state_dict['epoch']
    train_epoch(writer, train_loader, model, optimizer, scheduler, epoch)


def main(kwargs):
    # Workaround for pytorch large-scale multiprocessing issue
    # torch.multiprocessing.set_sharing_strategy('file_system')

    seed = 613
    torch.manual_seed(seed)
    np.random.seed(seed)

    debug_epoch(**kwargs)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        kwargs = dict(
            model_logdir='deep_tabular_HPO_TabERTransformer/acquirevaluedshopperschallenge_dfs_larger/TabERTransformer/Jan13_20-17-24-518277/xval0',
            checkpoint_id='best_auroc_71',
            device='cuda:0',
            num_workers=8
        )
    else:
        kwargs = pickle.loads(base64.b64decode(sys.argv[1]))
    main(kwargs)
