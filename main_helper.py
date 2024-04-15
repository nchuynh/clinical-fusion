# Import libraries
import sys
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn import metrics
import random
import json
from glob import glob
from collections import OrderedDict

# Import PyTorch
import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader

# Import custom modules for data loading, model definitions, loss calculations, and utility functions
import data_loader
import lstm, cnn
import myloss
import function
from utils import cal_metric

sys.path.append('./tools')
import py_op
import argparse

# Define function to parse command-line arguments and provide default arguments
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
    
    parser.add_argument('--use-unstructure', type=int, default=1, help='use unstructure')
    parser.add_argument('--csv-title', type=str, default='metrics.csv', help='csv title')

    parsed_args = parser.parse_args(args)
    
    parsed_args.data_dir = os.path.join(parsed_args.data_dir, 'processed')
    parsed_args.files_dir = os.path.join(parsed_args.data_dir, 'files')
    parsed_args.resample_dir = os.path.join(parsed_args.data_dir, 'resample_data')
    parsed_args.initial_dir = os.path.join(parsed_args.data_dir, 'initial_data')

    return parsed_args

# Define function to move tensor to the GPU
def _cuda(tensor, args, is_tensor=True):
    if args.gpu:
        if is_tensor:
            return tensor.cuda(non_blocking=True)
        else:
            return tensor.cuda()
    else:
        return tensor

# Define function to return learning rate
def get_lr(epoch, args):
    lr = args.lr
    return lr

# Definte function to map data into index and value based on the splitting
def index_value(data, args):
    if args.use_ve == 0:
        data = Variable(_cuda(data, args)) # [bs, 250]
        return data
    data = data.numpy()
    index = data / (args.split_num + 1)
    value = data % (args.split_num + 1)
    index = Variable(_cuda(torch.from_numpy(index.astype(np.int64)), args))
    value = Variable(_cuda(torch.from_numpy(value.astype(np.int64)), args))
    return [index, value]

# Define function for training and evaluating
def train_eval(data_loader, args, net, loss, epoch, optimizer, best_metric, phase='train'):
    print(phase)
    lr = get_lr(epoch, args)

    # If the phase is 'train', prepare model for training
    if phase == 'train':
        net.train()
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        net.eval()

    # Initialize lists to store loss, predictions and labels for performance evaluation
    loss_list, pred_list, label_list, = [], [], []

    # Iterate over batches of data provided by data_loader
    for b, data_list in enumerate(data_loader):
        data, dtime, demo, content, label, files = data_list
        if args.value_embedding == 'no':
            data = Variable(_cuda(data, args))
        else:
            data = index_value(data, args)

        dtime = Variable(_cuda(dtime, args)) 
        demo = Variable(_cuda(demo, args)) 
        content = Variable(_cuda(content, args)) 
        label = Variable(_cuda(label, args))

        # Process input through the model. Include clinical notes if use_unstructure is set
        if args.use_unstructure:
            output = net(data, dtime, demo, content) # [bs, 1]
        else:
            output = net(data, dtime, demo) # [bs, 1]

        # Compute loss for the current batch
        loss_output = loss(output, label)

        # Append current batch results to the lists
        pred_list.append(output.data.cpu().numpy())
        loss_list.append(loss_output[0].data.cpu().numpy())
        label_list.append(label.data.cpu().numpy())

        # If training, perform backpropagation and optimization step
        if phase == 'train':
            optimizer.zero_grad()
            loss_output[0].backward()
            optimizer.step()

    # Aggregate the predictions and labels
    pred = np.concatenate(pred_list, 0)
    label = np.concatenate(label_list, 0)

    # Calculate metrics
    if len(pred.shape) == 1:
        metric = function.compute_auc(label, pred)
    else:
        metrics = []
        auc_metrics = []
        for i_shape in range(pred.shape[1]):
            metric0 = cal_metric(label[:, i_shape], pred[:, i_shape])
            auc_metric = function.compute_auc(label[:, i_shape], pred[:, i_shape])
            print('AUC: {:3.4f}'.format(metric0[1]))
            metrics.append(metric0)
            auc_metrics.append(auc_metric)
        metric = np.mean(auc_metrics)
    avg_loss = np.mean(loss_list)

    print('LOSS: {:3.4f} \t'.format(avg_loss))

    # If phase is validation and the metric has improved, save the model
    if phase == 'valid' and best_metric[0] < metric:
        best_metric = [metric, epoch]
        function.save_model({'args': args, 'model': net, 'epoch':epoch, 'best_metric': best_metric})

    # If not training, print the best epoch and best metric
    if phase != 'train':
        print('BEST EPOCH: {:d}     BEST AUC: {:3.4f}'.format(best_metric[1], best_metric[0])) 

    return best_metric, avg_loss


def main(model, task, use_unstructure, phase, num_epochs=50):
    # Parse command-line or default arguments
    args = parse_arguments([])
    args.embed_size = 200
    args.hidden_size = args.rnn_size = args.embed_size 
    args.use_ve = 1
    args.n_visit = 24
    args.value_embedding = 'use_order'
    args.files_dir = args.files_dir
    args.data_dir = args.data_dir

    # Enable GPU if available
    if torch.cuda.is_available():
        args.gpu = 1
    else:
        args.gpu = 0

    # Model configurations and task parameters
    args.model = model
    args.task = task
    args.use_unstructure = use_unstructure
    args.phase = phase
    args.epochs = num_epochs
    args.csv_title = f'{args.model}_{args.task}_{args.use_unstructure}.csv'
    args.model_path = f'{args.model}_{args.task}_{args.use_unstructure}.ckpt'
    
    print ('\nMODEL: ', args.model)
    print ('TASK: ', args.task)
    print ('USE_UNSTRUCTURE: ', args.use_unstructure)
    if args.phase == 'train':
        print ('NUMBER OF EPOCHS: ', args.epochs)
    
    # Load additional configuration from JSON files
    args.n_ehr = len(json.load(open(os.path.join(args.files_dir, 'demo_index_dict.json'), 'r'))) + 10
    args.name_list = json.load(open(os.path.join(args.files_dir, 'feature_list.json'), 'r'))[1:]
    args.input_size = len(args.name_list)

    # Gather data files for each phase based on splits
    files = sorted(glob(os.path.join(args.data_dir, 'resample_data/*.csv')))
    data_splits = json.load(open(os.path.join(args.files_dir, 'splits.json'), 'r'))
    train_files = [f for idx in [0, 1, 2, 3, 4, 5, 6] for f in data_splits[idx]]
    valid_files = [f for idx in [7] for f in data_splits[idx]]
    test_files = [f for idx in [8, 9] for f in data_splits[idx]]

    # Adjust settings based on phase
    if args.phase == 'test':
        train_phase, valid_phase, test_phase, train_shuffle = 'test', 'test', 'test', False
    else:
        train_phase, valid_phase, test_phase, train_shuffle = 'train', 'valid', 'test', True

    # Create data loaders
    train_dataset = data_loader.DataBowl(args, train_files, phase=train_phase)
    valid_dataset = data_loader.DataBowl(args, valid_files, phase=valid_phase)
    test_dataset = data_loader.DataBowl(args, test_files, phase=test_phase)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=train_shuffle, num_workers=args.workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # Model setup
    args.vocab_size = args.input_size + 2
    if args.model == 'lstm':
        net = lstm.LSTM(args)
    else:
        net = cnn.CNN(args)

    # Initialize loss function
    loss = myloss.MultiClassLoss(0)

    # Move the model and loss function to GPU if available
    net = _cuda(net, args, 0)
    loss = _cuda(loss, args, 0)

    # Initialize metrics
    best_metric= [0,0]
    start_epoch = 0

    # Load model if resuming
    if args.resume:
        p_dict = {'model': net}
        function.load_model(p_dict, args.resume)
        best_metric = p_dict['best_metric']
        start_epoch = p_dict['epoch'] + 1

    # Initialize optimizer with parameters
    parameters_all = []
    for p in net.parameters():
        parameters_all.append(p)
    optimizer = torch.optim.Adam(parameters_all, args.lr)

    if args.phase == 'train':
        # Initialize dictionary to hold metrics
        dict_metrics = {'epoch': [], 'train_loss': [], 'train_auc': [], 'val_loss': [], 'val_auc': []}

        # Loop over each epoch
        for epoch in range(start_epoch, args.epochs):
            dict_metrics['epoch'].append(epoch + 1)
            print('\nSTART EPOCH: ', epoch)

            # Evaluate model on training data
            train_metric, train_loss = train_eval(train_loader, args, net, loss, epoch, optimizer, best_metric)
            dict_metrics['train_loss'].append(train_loss)
            dict_metrics['train_auc'].append(train_metric[0])

            # Evaluate model on validation data
            best_metric, val_loss = train_eval(valid_loader, args, net, loss, epoch, optimizer, best_metric, phase='valid')
            dict_metrics['val_loss'].append(val_loss)
            dict_metrics['val_auc'].append(best_metric[0])

        # Create dataframe from metrics dictionary
        df_metrics = pd.DataFrame(dict_metrics)
        # Save dataframe to CSV file
        df_metrics.to_csv(args.csv_title, index=False)

    # If phase is 'test', evaluate model on testing data
    elif args.phase == 'test':
        train_eval(test_loader, args, net, loss, 0, optimizer, best_metric, 'test')
