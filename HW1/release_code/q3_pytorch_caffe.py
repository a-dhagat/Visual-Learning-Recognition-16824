from __future__ import print_function

import numpy as np
import torch

import utils
# from q0_hello_mnist import SimpleCNN
from voc_dataset import VOCDataset

import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter


class CaffeNet(nn.Module):
    def __init__(self, num_classes=20, inp_size=227, c_dim=3):
        '''
        -------ConvLayer-------
        conv(kernel_size, stride, out_channels, padding)
        nn.conv2d(in_channels, out_channels, kernel_size, stride, padding)
        VALID: uses only valid input data if stride such that not all data can be used --> No padding
        SAME: keeps the output size same as the input size ------------------------------> Applies necessary padding to do that

        Conv Layer Accepts a volume of size W1×H1×D1
            Requires four hyperparameters:
                Number of filters K, their spatial extent F, the stride S, the amount of zero padding P.
            Produces a volume of size W2×H2×D2 where:
                W2=(W1−F+2P)/S+1
                H2=(H1−F+2P)/S+1 (i.e. width and height are computed equally by symmetry)
                D2=K

        -------MAXPOOL-------
        max_pool(kernel_size, stride)
        nn.MaxPool2d(kernel_size, stride)
        MaxPool Accepts a volume of size W1×H1×D1
            Requires two hyperparameters:
                their spatial extent F, the stride S,
            Produces a volume of size W2×H2×D2 where:
                W2=(W1−F)/S+1
                H2=(H1−F)/S+1
                D2=D1

        cite: http://cs231n.github.io/convolutional-networks/
        '''
        super().__init__()
        self.num_classes = num_classes
        # nn.conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv1 = nn.Conv2d(c_dim, 96, 11, 4, padding=0) # conv(11,4,96,'VALID')
        self.conv2 = nn.Conv2d(96, 256, 5, 1, padding=2) # conv(5,1,256,'SAME')
        self.conv3 = nn.Conv2d(256, 384, 3,1, padding=1) # conv(3,1,384,same)
        self.conv4 = nn.Conv2d(384, 384, 3, 1, padding=1) # conv(3,1,384,same)
        self.conv5 = nn.Conv2d(384, 256, 3, 1, padding=1)  # conv(3,1,256,same)
        self.non_linear = lambda x: F.relu(x, inplace=True)
        self.pool = nn.MaxPool2d(3,2)
        self.flat_dim = 256*6*6
        self.dropout = nn.Dropout(0.5,inplace=True)
        self.fc1 = nn.Sequential(*get_fc(self.flat_dim, 4096, 'relu'))
        self.fc2 = nn.Sequential(*get_fc(4096, 4096, 'relu'))
        self.fc3 = nn.Sequential(*get_fc(4096, 20, 'none'))

    def forward(self,x):
        """
        :param x: input image in shape of (N, C, H, W)
        :return: out: classification score in shape of (N, Nc)
        """
        N = x.size(0)
        x = self.conv1(x)
        x = self.non_linear(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.non_linear(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.non_linear(x)
        x = self.conv4(x)
        x = self.non_linear(x)
        x = self.conv5(x)
        x = self.non_linear(x)
        x = self.pool(x)
        flat_x = x.view(N, -1)
        out = self.fc1(flat_x)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.fc3(out)
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
    writer = SummaryWriter('../runs/q2_caffe_gam=0.9_changedTransform')
    # TODO: complete your dataloader in voc_dataset.py
    # import ipdb; ipdb.set_trace()
    train_loader = utils.get_data_loader('voc', train=True, batch_size=args.batch_size, split='trainval')
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test')

    # 2. define the model, and optimizer.
    # TODO: modify your model here!
    # bad idea of use simple CNN, but let's give it a shot!
    # In task 2, 3, 4, you might want to modify this line to be configurable to other models.
    # Remember: always reuse your code wisely.
    model = CaffeNet(num_classes=len(VOCDataset.CLASS_NAMES), inp_size=227, c_dim=3).to(device)
    model.train()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)
    cnt = 0
    loss_obj = nn.MultiLabelSoftMarginLoss()
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
            loss = loss_obj(output*wgt, wgt*target)
            # Calculate gradient w.r.t the loss
            loss.backward()
            # Optimizer takes one step
            optimizer.step()
            # Log info
            if epoch%10 == 0:
                torch.save(model.state_dict(), "../q2_caffe_models_v1/model_at_epoch_" + str(epoch) + ".pth")
            if cnt % args.log_every == 0:
                # todo: add your visualization code
                print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, cnt, 100. * batch_idx / len(train_loader), loss.item()))
                writer.add_scalar('Loss/train: ', loss.item(), cnt)
            # print('Batch idx: {} \r'.format(batch_idx), end='')
            # Validation iteration
            if cnt % args.val_every == 0:
                model.eval()
                # print("\n")
                ap, map = utils.eval_dataset_map(model, device, test_loader)
                writer.add_scalar('Validation mAP: ', map, cnt)
                model.train()
            cnt += 1

        scheduler.step()

    # Validation iteration
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test')
    ap, map = utils.eval_dataset_map(model, device, test_loader)

    print('----test-----')
    print(ap)
    print('mAP: ', map)
    writer.add_scalar('Testing mAP: ', map, cnt)
    writer.close()


if __name__ == '__main__':
    args, device = utils.parse_args()
    main()

