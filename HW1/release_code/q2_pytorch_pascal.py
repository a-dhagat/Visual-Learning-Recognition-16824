# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import numpy as np
import torch

import utils
# from q0_hello_mnist import SimpleCNN
from voc_dataset import VOCDataset

import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

class SimpleCNN(nn.Module):
    """
    Model definition
    """
    def __init__(self, num_classes=20, inp_size=256, c_dim=3):
        super().__init__()
        self.num_classes = num_classes
        # add your layer one by one -- one way to add layers
        self.conv1 = nn.Conv2d(c_dim, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        # TODO: Modify the code here
        self.nonlinear = lambda x: F.relu(x,inplace=True)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)

        # TODO: q0.1 Modify the code here
        self.flat_dim = 262144
        # chain your layers by Sequential -- another way
        # TODO: Modify the code here
        self.fc1 = nn.Sequential(*get_fc(self.flat_dim, 128, 'none'))
        
        # Changed activation from softmax to none
        self.fc2 = nn.Sequential(*get_fc(128, num_classes, 'none'))

    def forward(self, x):
        """
        :param x: input image in shape of (N, C, H, W)
        :return: out: classification score in shape of (N, Nc)
        """
        # import ipdb; ipdb.set_trace()
        N = x.size(0)
        x = self.conv1(x)
        x = self.nonlinear(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.nonlinear(x)
        x = self.pool2(x)

        # TODO: q0.1 hint (you might want to check the dimension of input here)
        # import ipdb; ipdb.set_trace()
        self.flat_dim = x.size(1)*x.size(2)*x.size(3)
        flat_x = x.view(N, self.flat_dim)
        out = self.fc1(flat_x)
        out = self.fc2(out)
        return out


def get_fc(inp_dim, out_dim, non_linear='relu'):
    """
    Mid-level API. It is useful to customize your own for large code repo.
    :param inp_dim: int, intput dimension
    :param out_dim: int, output dimension
    :param non_linear: str, 'relu', 'softmax'
    :return: list of layers [FC(inp_dim, out_dim), (non linear layer)]
    """
    layers = []
    layers.append(nn.Linear(inp_dim, out_dim))
    if non_linear == 'relu':
        layers.append(nn.ReLU())
    elif non_linear == 'softmax':
        layers.append(nn.Softmax(dim=1))
    elif non_linear == 'none':
        pass
    else:
        raise NotImplementedError
    return layers

def main():
    # TODO:  Initialize your visualizer here!
    # TODO: complete your dataloader in voc_dataset.py
    # import ipdb; ipdb.set_trace()
    train_loader = utils.get_data_loader('voc', train=True, batch_size=args.batch_size, split='trainval')
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test')

    # 2. define the model, and optimizer.
    # TODO: modify your model here!
    # bad idea of use simple CNN, but let's give it a shot!
    # In task 2, 3, 4, you might want to modify this line to be configurable to other models.
    # Remember: always reuse your code wisely.
    model = SimpleCNN(num_classes=len(VOCDataset.CLASS_NAMES), inp_size=20, c_dim=3).to(device)
    model.train()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)
    loss = nn.MultiLabelSoftMarginLoss()
    cnt = 0
    for epoch in range(args.epochs):

        for batch_idx, (data, target, wgt) in enumerate(train_loader):
            # Get a batch of data
            data, target, wgt = data.to(device), target.to(device), wgt.to(device)
            # import ipdb; ipdb.set_trace()
            optimizer.zero_grad()
            # Forward pass
            output = model(data)
            # Calculate the loss
            # TODO: your loss for multi-label clf?
            # loss = nn.BCELoss()
            # import ipdb; ipdb.set_trace()
            # output = torch.sigmoid(output)
            loss = loss(output*wgt, wgt*target)
            # Calculate gradient w.r.t the loss
            loss.backward()
            # Optimizer takes one step
            optimizer.step()
            # Log info
            if cnt % args.log_every == 0:
                # todo: add your visualization code
                print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, cnt, 100. * batch_idx / len(train_loader), loss.item()))
            # print('Batch idx: {} \r'.format(batch_idx), end='')
            # Validation iteration
            if cnt % args.val_every == 0:
                model.eval()
                # print("\n")
                ap, map = utils.eval_dataset_map(model, device, test_loader)
                model.train()
            cnt += 1

        scheduler.step()

    # Validation iteration
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test')
    ap, map = utils.eval_dataset_map(model, device, test_loader)

    print('----test-----')
    print(ap)
    print('mAP: ', map)


if __name__ == '__main__':
    args, device = utils.parse_args()
    main()