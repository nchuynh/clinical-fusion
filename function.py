#!/usr/bin/env python
# coding=utf-8
import numpy as np

import os
import torch
from sklearn import metrics

import argparse


def compute_nRMSE(pred, label, mask):
    '''
    same as 3dmice
    '''
    assert pred.shape == label.shape == mask.shape

    missing_indices = mask==1
    missing_pred = pred[missing_indices]
    missing_label = label[missing_indices]
    missing_rmse = np.sqrt(((missing_pred - missing_label) ** 2).mean())

    init_indices = mask==0
    init_pred = pred[init_indices]
    init_label = label[init_indices]
    init_rmse = np.sqrt(((init_pred - init_label) ** 2).mean())

    metric_list = [missing_rmse, init_rmse]
    for i in range(pred.shape[2]):
        apred = pred[:,:,i]
        alabel = label[:,:, i]
        amask = mask[:,:, i]

        mrmse, irmse = [], []
        for ip in range(len(apred)):
            ipred = apred[ip]
            ilabel = alabel[ip]
            imask = amask[ip]

            x = ilabel[imask>=0]
            if len(x) == 0:
                continue

            minv = ilabel[imask>=0].min()
            maxv = ilabel[imask>=0].max()
            if maxv == minv:
                continue

            init_indices = imask==0
            init_pred = ipred[init_indices]
            init_label = ilabel[init_indices]

            missing_indices = imask==1
            missing_pred = ipred[missing_indices]
            missing_label = ilabel[missing_indices]

            assert len(init_label) + len(missing_label) >= 2

            if len(init_pred) > 0:
                init_rmse = np.sqrt((((init_pred - init_label) / (maxv - minv)) ** 2).mean())
                irmse.append(init_rmse)

            if len(missing_pred) > 0:
                missing_rmse = np.sqrt((((missing_pred - missing_label)/ (maxv - minv)) ** 2).mean())
                mrmse.append(missing_rmse)

        metric_list.append(np.mean(mrmse))
        metric_list.append(np.mean(irmse))

    metric_list = np.array(metric_list)


    metric_list[0] = np.mean(metric_list[2:][::2])
    metric_list[1] = np.mean(metric_list[3:][::2])

    return metric_list


def save_model(p_dict):
    args = p_dict['args']
    model = p_dict['model']
    state_dict = model.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
    all_dict = {
            'epoch': p_dict['epoch'],
            'args': p_dict['args'],
            'best_metric': p_dict['best_metric'],
            'state_dict': state_dict 
            }
    torch.save(all_dict, args.model_path)

def load_model(p_dict, model_file):
    all_dict = torch.load(model_file)
    p_dict['epoch'] = all_dict['epoch']
    # p_dict['args'] = all_dict['args']
    p_dict['best_metric'] = all_dict['best_metric']
    # for k,v in all_dict['state_dict'].items():
    #     p_dict['model_dict'][k].load_state_dict(all_dict['state_dict'][k])
    p_dict['model'].load_state_dict(all_dict['state_dict'])

def compute_auc(labels, probs):
    fpr, tpr, thr = metrics.roc_curve(labels, probs)
    return metrics.auc(fpr, tpr)

def compute_metric(labels, probs):
    labels = np.array(labels)
    probs = np.array(probs)
    fpr, tpr, thresholds = metrics.roc_curve(labels, probs)
    auc = metrics.auc(fpr, tpr)
    aupr = metrics.average_precision_score(labels, probs)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    preds = [1 if prob >= optimal_threshold else 0 for prob in probs]
    tn, fp, fn, tp = metrics.confusion_matrix(labels, preds).ravel()
    precision = 1.0 * (tp / (tp + fp))
    sen = 1.0 * (tp / (tp + fn))  # recall
    spec = 1.0 * (tn / (tn + fp))
    f1 = metrics.f1_score(labels, preds)
    return precision, sen, spec, f1, auc, aupr

def parse_arguments(args=None):
    parser = argparse.ArgumentParser(description='clinical fusion help')
    
    parser.add_argument('--data-dir', type=str, default='./data/', help='selected and preprocessed data directory')
    parser.add_argument('--task', default='mortality', type=str, metavar='S', help='start from checkpoints')
    parser.add_argument('--last-time', metavar='last event time', type=int, default=-4, help='last time')
    parser.add_argument('--time-range', default=10000, type=int)
    parser.add_argument('--n-code', default=8, type=int, help='at most n codes for same visit')
    parser.add_argument('--n-visit', default=24, type=int, help='at most input n visits')
    parser.add_argument('--model', '-m', type=str, default='lstm', help='model')
    parser.add_argument('--split-num', metavar='split num', type=int, default=4000, help='split num')
    parser.add_argument('--split-nor', metavar='split normal range', type=int, default=200, help='split num')
    parser.add_argument('--use-glp', metavar='use global pooling operation', type=int, default=0, help='use global pooling operation')
    parser.add_argument('--use-value', metavar='use value embedding as input', type=int, default=1, help='use value embedding as input')
    parser.add_argument('--use-cat', metavar='use cat for time and value embedding', type=int, default=1, help='use cat or add')
    parser.add_argument('--embed-size', metavar='EMBED SIZE', type=int, default=512, help='embed size')
    parser.add_argument('--rnn-size', metavar='rnn SIZE', type=int, help='rnn size')
    parser.add_argument('--hidden-size', metavar='hidden SIZE', type=int, help='hidden size')
    parser.add_argument('--num-layers', metavar='num layers', type=int, default=2, help='num layers')
    parser.add_argument('--phase', default='train', type=str, help='train/test phase')
    parser.add_argument('--batch-size', '-b', metavar='BATCH SIZE', type=int, default=64, help='batch size')
    parser.add_argument('--model-path', type=str, default='models/best.ckpt', help='model path')
    parser.add_argument('--resume', default='', type=str, metavar='S', help='start from checkpoints')
    parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
    
    parsed_args = parser.parse_args(args)
    
    parsed_args.data_dir = os.path.join(parsed_args.data_dir, 'processed')
    parsed_args.files_dir = os.path.join(parsed_args.data_dir, 'files')
    parsed_args.resample_dir = os.path.join(parsed_args.data_dir, 'resample_data')
    parsed_args.initial_dir = os.path.join(parsed_args.data_dir, 'initial_data')

    return parsed_args
