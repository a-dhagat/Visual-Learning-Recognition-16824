from __future__ import print_function

import numpy as np
import torch

import utils
# from q0_hello_mnist import SimpleCNN
from voc_dataset import VOCDataset
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter


def main():
    # TODO:  Initialize your visualizer here!
    writer = SummaryWriter('../runs/q4_resnet18')
    # TODO: complete your dataloader in voc_dataset.py
    # import ipdb; ipdb.set_trace()
    train_loader = utils.get_data_loader('voc', train=True, batch_size=args.batch_size, split='trainval')
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test')

    # 2. define the model, and optimizer.
    # TODO: modify your model here!
    # bad idea of use simple CNN, but let's give it a shot!
    # In task 2, 3, 4, you might want to modify this line to be configurable to other models.
    # Remember: always reuse your code wisely.
    # model = CaffeNet(num_classes=len(VOCDataset.CLASS_NAMES), inp_size=227, c_dim=3).to(device)
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512,20)

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
            if cnt % args.log_every == 0:
                # todo: add your visualization code
                print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, cnt, 100. * batch_idx / len(train_loader), loss.item()))
                writer.add_scalar('Loss/train: ', loss.item(), cnt)
                if epoch%1 == 0:
                    torch.save(model.state_dict(), "../q4_resnet18_models/model_at_epoch_" + str(epoch) + ".pth")
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