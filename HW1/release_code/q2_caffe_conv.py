from __future__ import print_function

import numpy as np
import torch

import utils
# from q0_hello_mnist import SimpleCNN
from voc_dataset import VOCDataset

import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from q3_pytorch_caffe import CaffeNet

writer = SummaryWriter('../runs/q2_caffe_conv1_weight')

model = CaffeNet(num_classes=len(VOCDataset.CLASS_NAMES), inp_size=227, c_dim=3).to(device=0)
weight_list = []

num_epochs = [0,10,20,30,40]
for i in num_epochs:
    model.load_state_dict(torch.load('/home/biorobotics/VLR/Visual-Learning-Recognition-16824/HW1/q2_caffe_models_v1/model_at_epoch_' + str(i) + '.pth'))
    weight_list.append(model.conv1.weight)

iterations = [0,47,95]

for it in iterations:
    for epoch, weight in enumerate(weight_list):
        writer.add_image("epoch"+str(epoch)+"_iter"+str(it), weight[it])
    
writer.close()