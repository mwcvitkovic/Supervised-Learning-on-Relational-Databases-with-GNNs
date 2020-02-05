import base64
import json
import os
import pickle
import pprint
import sys

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.optim as opt
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, log_loss, accuracy_score
from torch.optim.lr_scheduler import CyclicLR, OneCycleLR, ExponentialLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
from tqdm import tqdm

import models
from data.DatabaseDataset import DatabaseDataset
from data.TabularDataset import TabularDataset
from data.utils import write_kaggle_submission_file
from models.GNN.GNNModelBase import GNNModelBase
from models.tabular.TabModelBase import TabModelBase
from models.utils import save_train_kwargs, recursive_to, save_model_checkpoint, get_good_lr, register_module_hooks
from utils import setup_writer, get_train_val_test_datasets, get_dataloader, log_param_values, \
    format_hparam_dict_for_tb, model_to_device, get_optim_with_correct_wd


def train_epoch(writer, train_loader, model, optimizer, scheduler, epoch):
    model.train()
    writer.batches_done = epoch * len(train_loader)
    # t = time.perf_counter()
    for batch_idx, (input, label) in enumerate(tqdm(train_loader)):
        if batch_idx == 0:
            writer.add_scalar('Training/Learning Rate', optimizer.param_groups[0]['lr'], writer.batches_done)
        # writer.add_scalar('CodeProfiling/Train/Batch Size', bdgl.number_of_nodes(), writer.batches_done)
        # writer.add_scalar('CodeProfiling/Train/Get Batch Time', time.perf_counter() - t, writer.batches_done)
        # t = time.perf_counter()
        recursive_to((input, label), model.device)
        optimizer.zero_grad()
        # writer.add_scalar('CodeProfiling/Train/Load to GPU Time', time.perf_counter() - t, writer.batches_done)
        # t = time.perf_counter()
        output = model(input)
        # writer.add_scalar('CodeProfiling/Train/Compute Output Time', time.perf_counter() - t, writer.batches_done)
        # t = time.perf_counter()
        loss = model.loss_fxn(output, label)
        if torch.isnan(loss):
            raise ValueError('Loss was NaN')
        # writer.add_scalar('CodeProfiling/Compute Loss Time', time.perf_counter() - t, writer.batches_done)
        # t = time.perf_counter()
        loss.backward()
        # writer.add_scalar('CodeProfiling/Backward Time', time.perf_counter() - t, writer.batches_done)
        # t = time.perf_counter()
        optimizer.step()
        # writer.add_scalar('CodeProfiling/Train/Grad Step Time', time.perf_counter() - t, writer.batches_done)
        # t = time.perf_counter()
        if batch_idx == 0:
            writer.add_scalar('Training/{}'.format(model.loss_fxn.__class__.__name__), loss.item(),
                              writer.batches_done)
        writer.batches_done += 1
        # writer.add_scalar('CodeProfiling/Other stuff Time', time.perf_counter() - t, writer.batches_done)
        # t = time.perf_counter()
        log_param_values(writer, model)
        if isinstance(scheduler, (CyclicLR, OneCycleLR)):
            scheduler.step()
    if isinstance(scheduler, (ExponentialLR, CosineAnnealingWarmRestarts)):
        scheduler.step()


def validate_model(writer, val_loader, model, epoch):
    model.eval()
    with torch.autograd.no_grad():
        val_loss = torch.Tensor([0])
        n_correct = torch.Tensor([0])
        labels = []
        probs = []
        for batch_idx, (input, label) in enumerate(tqdm(val_loader)):
            recursive_to((input, label), model.device)
            output = model(input)
            val_loss += model.loss_fxn(output, label).cpu()  # sum up mean batch losses
            if isinstance(output, torch.Tensor):
                probs.append(torch.softmax(output, dim=1).cpu())
                pred = model.pred_from_output(output)
                n_correct += pred.eq(label.view_as(pred)).sum().cpu()
                labels.append(label.cpu())

        val_loss = (val_loss.cpu() / len(val_loader)).item()
        writer.add_scalar('Validation/{}'.format(model.loss_fxn.__class__.__name__), val_loss, writer.batches_done)
        print(f'val_loss epoch {epoch}: {val_loss}')

        if isinstance(output, torch.Tensor):
            labels = torch.cat(labels, dim=0).cpu().numpy()
            probs = torch.cat(probs, dim=0).cpu().numpy()

            val_acc = (100 * n_correct / len(val_loader.dataset)).item()
            writer.add_scalar('Validation/Accuracy', val_acc, writer.batches_done)
            print(f'val_acc epoch {epoch}: {val_acc}')

            val_auroc = roc_auc_score(labels, probs[:, 1])
            writer.add_scalar('Validation/AUROC Score', val_auroc, writer.batches_done)
            print(f'val_auroc epoch {epoch}: {val_auroc}')

            plot_validation_info(writer, labels, probs)
        else:
            val_auroc = val_acc = None

        return val_auroc, val_acc, val_loss


def plot_validation_info(writer, labels, probs):
    # Plot predictive distributions
    writer.add_histogram('Validation/Negative Probs', probs[:, 0], writer.batches_done)
    writer.add_histogram('Validation/Positive Probs', probs[:, 1], writer.batches_done)

    writer.add_pr_curve('Validation/PR Curve', labels, probs[:, 1], writer.batches_done)

    distpos = pd.Series(probs[:, 1][labels.nonzero()], name='True Positive')
    distneg = pd.Series(probs[:, 1][(labels == 0).nonzero()], name='True Negative')
    sns.distplot(distpos, label='True Positive', norm_hist=False, kde=False, color='b')
    plt.legend()
    plt.xlabel('Predicted Probability')
    plt.xlim(0, 1)
    plt.ylabel('Count')
    writer.add_figure('Validation/True Positive Predictive Distribution', plt.gcf(), writer.batches_done)
    sns.distplot(distneg, label='True Negative', norm_hist=False, kde=False, color='y')
    plt.legend()
    plt.xlabel('Predicted Probability')
    plt.xlim(0, 1)
    plt.ylabel('Count')
    writer.add_figure('Validation/True Negative Predictive Distribution', plt.gcf(), writer.batches_done)
    sns.distplot(distneg, label='True Negative', norm_hist=False, kde=False, color='y')
    sns.distplot(distpos, label='True Positive', norm_hist=False, kde=False, color='b')
    plt.legend()
    plt.xlabel('Predicted Probability')
    plt.xlim(0, 1)
    plt.ylabel('Count')
    writer.add_figure('Validation/Predictive Distribution', plt.gcf(), writer.batches_done)

    # Plot ROC
    val_auroc = roc_auc_score(labels, probs[:, 1])
    fpr, tpr, _ = roc_curve(labels, probs[:, 1], pos_label=1)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (area = {val_auroc})')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    writer.add_figure('Validation/ROC Curve', plt.gcf(), writer.batches_done)

    # Plot confusion matrix
    preds = np.argmax(probs, axis=1)
    cm = confusion_matrix(labels, preds)
    df_cm = pd.DataFrame(cm, columns=np.unique(labels), index=np.unique(labels))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize=(10, 7))
    sns.set(font_scale=1.4)
    sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt='g')
    plt.yticks((0, 0.5, 0.5, 1.5, 2.0), ('', 0, '', 1, ''))  # dumb hack for weird sns plotting
    writer.add_figure('Validation/Confusion Matrix', plt.gcf(), writer.batches_done)


def train_model(
        writer,
        seed,
        log_dir,
        debug_network,
        dataset_name,
        train_test_split,
        encoders,
        max_nodes_per_graph,
        train_fraction_to_use,
        sampler_class_name,
        sampler_class_kwargs,
        model_class_name,
        model_kwargs,
        batch_size,
        epochs,
        optimizer_class_name,
        optimizer_kwargs,
        lr_scheduler_class_name,
        lr_scheduler_kwargs,
        early_stopping_patience,
        wd_bias,
        wd_embed,
        wd_bn,
        load_model_weights_from='',
        early_stopping_metric='loss',
        device='cpu',
        num_workers=0,
        find_lr=True):
    train_data, val_data, _ = get_train_val_test_datasets(dataset_name=dataset_name,
                                                          train_test_split=train_test_split,
                                                          encoders=encoders,
                                                          train_fraction_to_use=train_fraction_to_use)
    train_loader = get_dataloader(dataset=train_data,
                                  batch_size=batch_size,
                                  sampler_class_name=sampler_class_name,
                                  sampler_class_kwargs=sampler_class_kwargs,
                                  num_workers=num_workers,
                                  max_nodes_per_graph=max_nodes_per_graph)
    print(f'Batches per train epoch: {len(train_loader)}')
    print(f'Total batches: {len(train_loader) * epochs}')
    val_loader = get_dataloader(dataset=val_data,
                                batch_size=batch_size,
                                sampler_class_name='SequentialSampler',
                                num_workers=num_workers,
                                max_nodes_per_graph=max_nodes_per_graph)

    def init_model():
        model_class = models.__dict__[model_class_name]
        if isinstance(train_data, TabularDataset):
            assert issubclass(model_class, TabModelBase)
            model_kwargs.update(
                n_cont_features=train_data.n_cont_features,
                cat_feat_origin_cards=train_data.cat_feat_origin_cards
            )
        elif isinstance(train_data, DatabaseDataset):
            assert issubclass(model_class, GNNModelBase)
            model_kwargs.update(
                feature_encoders=train_data.feature_encoders
            )
        else:
            raise ValueError
        model = model_class(writer=writer,
                            dataset_name=dataset_name,
                            **model_kwargs)
        if load_model_weights_from:
            state_dict = torch.load(load_model_weights_from, map_location=torch.device('cpu'))
            retval = model.load_state_dict(state_dict['model'], strict=False)
            print(f'Missing modules:\n{pprint.pformat(retval.missing_keys)}')
            print(f'Unexpected modules:\n{pprint.pformat(retval.unexpected_keys)}')
        model_to_device(model, device)

        # If debugging, add hooks to all modules
        if debug_network:
            register_module_hooks('model', model, writer)

        return model

    # Optionally find good learning rate
    if find_lr:
        print('Finding good learning rate')
        model = init_model()
        optimizer = get_optim_with_correct_wd(optimizer_class_name, model, optimizer_kwargs, wd_bias, wd_embed, wd_bn)
        good_lr = get_good_lr(model, optimizer, train_loader, init_value=1e-7, final_value=1.0, beta=0.98)
        optimizer_kwargs.update(lr=good_lr)
        writer.train_kwargs['optimizer_kwargs'].update(lr=good_lr)
        if lr_scheduler_class_name == 'CyclicLR':
            lr_scheduler_kwargs.update(max_lr=good_lr, base_lr=good_lr / 100)
            writer.train_kwargs['lr_scheduler_kwargs'].update(max_lr=good_lr, base_lr=good_lr / 100)
        elif lr_scheduler_class_name == 'OneCycleLR':
            lr_scheduler_kwargs.update(max_lr=good_lr)
            writer.train_kwargs['lr_scheduler_kwargs'].update(max_lr=good_lr)

    model = init_model()
    optimizer = get_optim_with_correct_wd(optimizer_class_name, model, optimizer_kwargs, wd_bias, wd_embed, wd_bn)
    scheduler = opt.lr_scheduler.__dict__[lr_scheduler_class_name](optimizer, **lr_scheduler_kwargs)

    # Run train loop with early stopping
    best_auroc = -1
    best_acc = -1
    best_loss = np.inf
    best_epoch = -1
    try:
        for epoch in tqdm(range(epochs)):
            print(f'Epoch: {epoch}')
            log_param_values(writer, model)
            if epoch % 20 == 0:
                save_model_checkpoint(writer, epoch, model, optimizer, scheduler)
            val_auroc, val_acc, val_loss = validate_model(writer, val_loader, model, epoch)
            best = False
            if val_auroc is not None and val_auroc > best_auroc:
                best_auroc = val_auroc
                save_model_checkpoint(writer, epoch, model, optimizer, scheduler, chkpt_name='best_auroc')
                if early_stopping_metric == 'auroc':
                    best = True
            if val_acc is not None and val_acc > best_acc:
                best_acc = val_acc
                save_model_checkpoint(writer, epoch, model, optimizer, scheduler, chkpt_name='best_acc')
                if early_stopping_metric == 'acc':
                    best = True
            if val_loss < best_loss:
                best_loss = val_loss
                save_model_checkpoint(writer, epoch, model, optimizer, scheduler, chkpt_name='best_loss')
                if early_stopping_metric == 'loss':
                    best = True
            if early_stopping_metric == 'auroc':
                m = val_auroc
            elif early_stopping_metric == 'acc':
                m = val_acc
            elif early_stopping_metric == 'loss':
                m = -1 * val_loss
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(m)
            if best:
                best_epoch = epoch
            if epoch - best_epoch >= early_stopping_patience:
                Path(os.path.join(writer.log_dir, 'stopped_early.info')).touch()
                break
            train_epoch(writer, train_loader, model, optimizer, scheduler, epoch)
            if hasattr(model, 'prune'):
                model.prune(epoch, m)
        else:
            save_model_checkpoint(writer, epoch, model, optimizer, scheduler)
            validate_model(writer, val_loader, model, epoch)
            Path(os.path.join(writer.log_dir, 'finished_all_epochs.info')).touch()
        writer.add_hparams(format_hparam_dict_for_tb(writer.train_kwargs), {'hparam/best_auroc': best_auroc,
                                                                            'hparam/best_acc': best_acc,
                                                                            'hparam/best_loss': best_loss,
                                                                            'hparam/best_epoch': best_epoch})
    except Exception as e:
        Path(os.path.join(writer.log_dir, 'failed.info')).touch()
        writer.add_hparams(format_hparam_dict_for_tb(writer.train_kwargs), {'hparam/best_auroc': best_auroc,
                                                                            'hparam/best_acc': best_acc,
                                                                            'hparam/best_loss': best_loss,
                                                                            'hparam/best_epoch': best_epoch})
        raise e


def train_non_deep_model(writer,
                         seed,
                         log_dir,
                         debug_network,
                         dataset_name,
                         train_test_split,
                         encoders,
                         train_fraction_to_use,
                         model_class_name,
                         model_kwargs,
                         num_boost_round,
                         early_stopping_patience,
                         num_workers=-1):
    train_data, val_data, orig_test_data = get_train_val_test_datasets(dataset_name=dataset_name,
                                                                       train_test_split=train_test_split,
                                                                       encoders=encoders,
                                                                       train_fraction_to_use=train_fraction_to_use)

    categorical_feature = [i[0] for i in train_data.cat_feat_origin_cards]
    feature_names = categorical_feature + train_data.cont_feat_origin
    train_data = lgb.Dataset(
        np.concatenate([t.numpy() for t in [train_data.cat_data, train_data.cont_data] if t is not None], axis=1),
        label=train_data.targets.numpy(),
        feature_name=feature_names,
        categorical_feature=categorical_feature
    )
    raw_val_data = np.concatenate([t.numpy() for t in [val_data.cat_data, val_data.cont_data] if t is not None], axis=1)
    val_data = lgb.Dataset(raw_val_data,
                           label=val_data.targets.numpy(),
                           feature_name=feature_names,
                           categorical_feature=categorical_feature,
                           reference=train_data
                           )
    test_labels = orig_test_data.targets.numpy().T
    test_data = np.concatenate(
        [t.numpy() for t in [orig_test_data.cat_data, orig_test_data.cont_data] if t is not None], axis=1)

    try:
        # Train
        param = {'num_leaves': model_kwargs['num_leaves'],
                 'min_data_in_leaf': model_kwargs['min_data_in_leaf'],
                 'objective': 'binary',
                 'n_jobs': num_workers,
                 'metric': ['cross_entropy', 'binary_error', 'auc'],
                 'first_metric_only': True,
                 }
        bst = lgb.train(
            param,
            train_data,
            valid_sets=[val_data],
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_patience
        )

        # Plot val metrics
        if bst.best_iteration < num_boost_round:
            Path(os.path.join(writer.log_dir, 'stopped_early.info')).touch()
        else:
            Path(os.path.join(writer.log_dir, 'finished_all_epochs.info')).touch()
        bst.save_model(os.path.join(writer.log_dir, 'model_checkpoint_best_val_loss.lgb'),
                       num_iteration=bst.best_iteration)
        val_probs = bst.predict(raw_val_data, num_iteration=bst.best_iteration)
        val_probs = np.vstack([1 - val_probs, val_probs]).T
        plot_validation_info(writer, val_data.label, val_probs)
        writer.add_hparams(format_hparam_dict_for_tb(writer.train_kwargs),
                           {'hparam/best_auroc': bst.best_score['valid_0']['auc'],
                            'hparam/best_acc': 100 * (1 - bst.best_score['valid_0']['binary_error']),
                            'hparam/best_loss': bst.best_score['valid_0']['cross_entropy'],
                            'hparam/best_iter': bst.best_iteration})

        # Save test metrics
        for checkpoint_id in ['best_loss', 'best_auroc', 'best_acc']:
            test_probs = bst.predict(test_data, num_iteration=bst.best_iteration)
            test_probs = np.vstack([1 - test_probs, test_probs]).T
            results_dir = os.path.join(writer.log_dir, 'evaluations', f'model_checkpoint_{checkpoint_id}')
            os.makedirs(results_dir, exist_ok=True)
            results_file = os.path.join(results_dir, 'results.json')
            if train_test_split == 'use_full_train':
                # Write kaggle submission file
                test_probs = test_probs[:, 1]
                test_ids = orig_test_data.datapoint_ids
                predictions = pd.DataFrame({'dp_id': test_ids, 'prob': test_probs})
                prediction_file = os.path.join(results_dir, 'kaggle_submission.csv')
                write_kaggle_submission_file(dataset_name, predictions, prediction_file)
            else:
                results = {'test_loss': log_loss(test_labels, test_probs),
                           'test_accuracy': 100 * accuracy_score(test_labels, test_probs.argmax(axis=1)),
                           'test_auroc': roc_auc_score(test_labels, test_probs[:, 1])}
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)

    except Exception as e:
        Path(os.path.join(writer.log_dir, 'failed.info')).touch()
        writer.add_hparams(format_hparam_dict_for_tb(writer.train_kwargs), {'hparam/best_auroc': -1,
                                                                            'hparam/best_acc': -1,
                                                                            'hparam/best_loss': np.inf,
                                                                            'hparam/best_iter': -1})
        raise e


def main(kwargs):
    # Workaround for pytorch large-scale multiprocessing issue, if you're using a lot of dataloaders
    # torch.multiprocessing.set_sharing_strategy('file_system')

    torch.manual_seed(kwargs['seed'])
    np.random.seed(kwargs['seed'])

    writer = setup_writer(kwargs['log_dir'], kwargs['debug_network'])
    save_train_kwargs(writer, kwargs)
    writer.add_text('train_kwargs', pprint.pformat(kwargs).replace('\n', '\t\n'))
    writer.train_kwargs = kwargs

    if kwargs['model_class_name'] == 'LightGBM':
        train_non_deep_model(writer, **kwargs)
    else:
        train_model(writer, **kwargs)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        kwargs = dict()
        # # This is here as an example:
        # kwargs = {
        #     "seed": 1234,
        #     "debug_network": False,
        #     "encoders": {
        #         "CATEGORICAL": "CategoricalOrdinalEnc",
        #         "SCALAR": "ScalarRobustScalerEnc",
        #         "DATETIME": "DatetimeScalarEnc",
        #         "LATLONG": "LatLongScalarEnc",
        #         "TEXT": "TextSummaryScalarEnc"
        #     },
        #     "early_stopping_patience": 250,
        #     "early_stopping_metric": "loss",
        #     "max_nodes_per_graph": False,
        #     "train_fraction_to_use": 1.0,
        #     "dataset_name": "acquirevaluedshopperschallenge_main_table",
        #     "device": "cuda",
        #     "find_lr": False,
        #     "epochs": 1000,
        #     "batch_size": 1024,
        #     "num_workers": 8,
        #     "lr_scheduler_class_name": "StepLR",
        #     "lr_scheduler_kwargs": {
        #         "step_size": 1,
        #         "gamma": 1.0
        #     },
        #     "optimizer_class_name": "AdamW",
        #     "optimizer_kwargs": {
        #         "lr": 0.0001,
        #         "weight_decay": 0.01
        #     },
        #     "wd_bias": False,
        #     "wd_embed": False,
        #     "wd_bn": False,
        #     "sampler_class_name": "RandomSampler",
        #     "sampler_class_kwargs": {},
        #     "model_class_name": "TabMLP",
        #     "model_kwargs": {
        #         "layer_sizes": [
        #             136,
        #             34
        #         ],
        #         "max_emb_dim": 32,
        #         "p_dropout": 0.0,
        #         "one_hot_embeddings": True,
        #         "drop_whole_embeddings": False,
        #         "activation_class_name": "SELU",
        #         "activation_class_kwargs": {},
        #         "norm_class_name": "BatchNorm1d",
        #         "norm_class_kwargs": {},
        #         "loss_class_name": "CrossEntropyLoss",
        #         "loss_class_kwargs": {}
        #     },
        #     "train_test_split": "xval0",
        #     "load_model_weights_from": "",
        #     "log_dir": "debug"
        # }
    else:
        kwargs = pickle.loads(base64.b64decode(sys.argv[1]))
    main(kwargs)
