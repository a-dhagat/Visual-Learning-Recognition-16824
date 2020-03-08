# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import argparse

import numpy as np
import os
import sklearn.metrics

import torch
from torch.utils.data import DataLoader

def parse_args():
    """
    :return:  args: experiment configs, device: use CUDA or cpu
    """
    parser = argparse.ArgumentParser(description='Assignment 1')
    # config for dataset

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log_every', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--val_every', type=int, default=100, metavar='N',
                        help='how many batches to wait before evaluating model')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--flag', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    return args, device


def get_data_loader(name='voc', train=True, batch_size=64, split='train'):
    flag = 1

    if name == 'voc':
        from voc_dataset import VOCDataset
        dataset = VOCDataset(split, 227)
    else:
        raise NotImplementedError

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
    )
    if (flag==1):
        return loader
    
    else:
        return loader, dataset.index_list


def compute_ap(gt, pred, valid, average=None):
    """
    Compute the multi-label classification accuracy.
    Args:
        gt (np.ndarray): Shape Nx20, 0 or 1, 1 if the object i is present in that
            image.
        pred (np.ndarray): Shape Nx20, probability of that object in the image
            (output probablitiy).
        valid (np.ndarray): Shape Nx20, 0 if you want to ignore that class for that
            image. Some objects are labeled as ambiguous.
    Returns:
        AP (list): average precision for all classes
    """
    # import ipdb; ipdb.set_trace()
    nclasses = gt.shape[1]
    AP = []
    for cid in range(nclasses):
        gt_cls = gt[:, cid][valid[:, cid] > 0].astype('float32')
        pred_cls = pred[:, cid][valid[:, cid] > 0].astype('float32')
        # As per PhilK. code:
        # https://github.com/philkr/voc-classification/blob/master/src/train_cls.py
        pred_cls -= 1e-5 * gt_cls
        # print(gt_cls)
        # print(pred_cls)
        # quit()
        ap = sklearn.metrics.average_precision_score(
            gt_cls, pred_cls, average=average)
        AP.append(ap)
    return AP


def eval_dataset_map(model, device, test_loader):
    """
    Evaluate the model with the given dataset
    Args:
         model (keras.Model): model to be evaluated
         dataset (tf.data.Dataset): evaluation dataset
    Returns:
         AP (list): Average Precision for all classes
         MAP (float): mean average precision
    """
    AP = []
    pred_list = []
    target_list = []
    wgt_list = []
    with torch.no_grad():
        for idx, (data, target, wgt) in enumerate(test_loader):
            ## TODO insert your code here
            
            pred = model(data.to(device))
            # pred = torch.sigmoid(pred)
            # print(np.sum(target.detach().cpu().numpy()))
            pred_list.append(pred.detach().cpu().numpy())
            target_list.append(target.detach().cpu().numpy())
            wgt_list.append(wgt.detach().cpu().numpy())
            # ap = compute_ap(target.detach().cpu().numpy(), pred.detach().cpu().numpy(), wgt.detach().cpu().numpy())
            # AP.append(ap)
            # print("AP: ", AP)
            # quit()
        AP = compute_ap(np.concatenate(target_list), np.concatenate(pred_list), np.concatenate(wgt_list))
        # print("AP: ", AP)
        # AP_List = AP_List + AP
            # pass
        # compute_ap()
    # AP = compute_ap(gt, pred, valid)
    # AP = AP[1:]
        mAP = np.mean(AP)
    return AP, mAP

