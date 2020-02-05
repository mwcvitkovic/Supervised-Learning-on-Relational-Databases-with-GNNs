import base64
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import models
from data.DatabaseDataset import DatabaseDataset
from data.TabularDataset import TabularDataset
from data.utils import write_kaggle_submission_file
from models.GNN.GNNModelBase import GNNModelBase
from models.tabular.TabModelBase import TabModelBase
from models.utils import recursive_to
from utils import DummyWriter, get_train_val_test_datasets, get_dataloader, model_to_device, get_train_test_dp_ids


def evaluate_model(test_loader, train_kwargs, results_dir, model):
    train_test_split = train_kwargs['train_test_split']
    db_name = train_kwargs['dataset_name']
    model.eval()
    with torch.autograd.no_grad():
        probs = []
        if train_test_split != 'use_full_train':
            test_loss = torch.Tensor([0])
            n_correct = torch.Tensor([0])
            labels = []

        for batch_idx, (input, label) in enumerate(tqdm(test_loader)):
            recursive_to((input, label), model.device)
            output = model(input)
            probs.append(torch.softmax(output, dim=1).cpu())

            if train_test_split != 'use_full_train':
                pred = model.pred_from_output(output)
                test_loss += model.loss_fxn(output, label).cpu()  # sum up mean batch losses
                n_correct += pred.eq(label.view_as(pred)).sum().cpu()
                labels.append(label.cpu())
        probs = torch.cat(probs, dim=0).cpu()

        if train_test_split == 'use_full_train':
            # Write kaggle submission file
            test_probs = probs[:, 1]
            test_ids = test_loader.dataset.datapoint_ids
            predictions = pd.DataFrame({'dp_id': test_ids, 'prob': test_probs})
            prediction_file = os.path.join(results_dir, 'kaggle_submission.csv')
            write_kaggle_submission_file(db_name, predictions, prediction_file)

        # Write test results, if cross validating
        if train_test_split != 'use_full_train':
            results = {}

            test_loss = test_loss.cpu() / len(test_loader)
            results['test_loss'] = test_loss.item()

            test_acc = 100 * n_correct / len(test_loader.dataset)
            results['test_accuracy'] = test_acc.item()

            labels = torch.cat(labels, dim=0).cpu()
            test_auroc = roc_auc_score(labels, probs[:, 1])
            results['test_auroc'] = test_auroc

            results_file = os.path.join(results_dir, 'results.json')
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)


def dump_activations(ds_name, train_kwargs, train_data, encoders, results_dir, model,
                     module_acts_to_dump, num_workers):
    # We dump activations for every datapoint here, even ones that weren't in model's train, val, or test
    train_dp_ids, test_dp_ids = get_train_test_dp_ids(ds_name)
    dp_ids = np.concatenate([train_dp_ids, test_dp_ids]) if test_dp_ids is not None else train_dp_ids

    if ds_name in ['acquirevaluedshopperschallenge', 'homecreditdefaultrisk', 'kddcup2014']:
        dataset = DatabaseDataset(ds_name, dp_ids, encoders)
    else:
        dataset = TabularDataset(ds_name, dp_ids, encoders)
        dataset.encode(train_data.feature_encoders)
    loader = get_dataloader(dataset=dataset,
                            batch_size=train_kwargs['batch_size'],
                            sampler_class_name='SequentialSampler',
                            num_workers=num_workers,
                            max_nodes_per_graph=train_kwargs['max_nodes_per_graph'])

    model.eval()
    acts = []

    def save_acts(module, input, output):
        acts.append(input[0].detach().cpu().numpy())

    module = eval(f'model.{module_acts_to_dump}')
    module.register_forward_hook(save_acts)
    with torch.autograd.no_grad():
        for batch_idx, (input, label) in enumerate(tqdm(loader)):
            recursive_to((input, label), model.device)
            model(input)
    acts = np.concatenate(acts, axis=0)
    np.save(os.path.join(results_dir, f'{module_acts_to_dump}.activations'), acts)

    return acts


def start_evaluating(
        do_evaluate,
        do_dump_activations,
        module_acts_to_dump,
        model_logdir,
        checkpoint_id,
        device,
        num_workers):
    with open(os.path.join(model_logdir, 'train_kwargs.json')) as f:
        train_kwargs = json.load(f)
    ds_name = train_kwargs['dataset_name']
    encoders = train_kwargs['encoders']
    train_data, _, test_data = get_train_val_test_datasets(dataset_name=ds_name,
                                                           train_test_split=train_kwargs['train_test_split'],
                                                           encoders=encoders,
                                                           train_fraction_to_use=train_kwargs.get(
                                                               'train_fraction_to_use', 1.0))
    test_loader = get_dataloader(dataset=test_data,
                                 batch_size=train_kwargs['batch_size'],
                                 sampler_class_name='SequentialSampler',
                                 num_workers=num_workers,
                                 max_nodes_per_graph=train_kwargs['max_nodes_per_graph'])
    writer = DummyWriter()
    model_class = models.__dict__[train_kwargs['model_class_name']]
    if isinstance(train_data, TabularDataset):
        assert issubclass(model_class, TabModelBase)
        train_kwargs['model_kwargs'].update(
            n_cont_features=train_data.n_cont_features,
            cat_feat_origin_cards=train_data.cat_feat_origin_cards
        )
    elif isinstance(train_data, DatabaseDataset):
        assert issubclass(model_class, GNNModelBase)
        train_kwargs['model_kwargs'].update(
            feature_encoders=train_data.feature_encoders
        )
    else:
        raise ValueError
    model = model_class(writer=writer,
                        dataset_name=train_kwargs['dataset_name'],
                        **train_kwargs['model_kwargs'])
    if 'best' in checkpoint_id:
        checkpoint_path = [f for f in os.listdir(model_logdir) if checkpoint_id in f]
        assert len(checkpoint_path) == 1, 'Wrong number of best checkpoints'
        checkpoint_path = os.path.join(model_logdir, checkpoint_path[0])
    else:
        checkpoint_path = os.path.join(model_logdir, f'model_checkpoint_{checkpoint_id}.pt')
    if torch.cuda.is_available() and 'cuda' in device:
        state_dict = torch.load(checkpoint_path, map_location=torch.device(device))
    else:
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['model'])
    model_to_device(model, device)

    results_dir = os.path.join(model_logdir, 'evaluations', f'model_checkpoint_{checkpoint_id}')
    os.makedirs(results_dir, exist_ok=True)

    if do_evaluate:
        evaluate_model(test_loader, train_kwargs, results_dir, model)

    if do_dump_activations:
        acts = dump_activations(ds_name, train_kwargs, train_data, encoders, results_dir, model, module_acts_to_dump,
                                num_workers)
        return acts


def main(kwargs):
    # Workaround for pytorch large-scale multiprocessing issue
    # torch.multiprocessing.set_sharing_strategy('file_system')

    seed = 612
    torch.manual_seed(seed)
    np.random.seed(seed)

    start_evaluating(**kwargs)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        kwargs = dict()
        # # This is here as an example:
        # kwargs = dict(
        #     do_evaluate=False,
        #     do_dump_activations=True,
        #     module_acts_to_dump="fcout[-1]",
        #     model_logdir='',
        #     checkpoint_id='best_auroc',
        #     device='cuda:0',
        #     num_workers=8
        # )
    else:
        kwargs = pickle.loads(base64.b64decode(sys.argv[1]))
    main(kwargs)
