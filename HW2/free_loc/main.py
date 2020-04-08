import argparse
import os
import shutil
import time
import sys
sys.path.insert(0, '../faster_rcnn')
# import sklearn
# import sklearn.metrics

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import pdb

from datasets.factory import get_imdb
from custom import *

from tensorboardX import SummaryWriter

val_flag = True

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

                    

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', default='localizer_alexnet_robust')
parser.add_argument(
    '-j',
    '--workers',
    default=4,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 4)')
parser.add_argument(
    '--epochs',
    default=30,
    type=int,
    metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '-b',
    '--batch-size',
    default=256,
    type=int,
    metavar='N',
    help='mini-batch size (default: 256)')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.1,
    type=float,
    metavar='LR',
    help='initial learning rate')
parser.add_argument(
    '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
    '--weight-decay',
    '--wd',
    default=1e-4,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-4)')
parser.add_argument(
    '--print-freq',
    '-p',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--eval-freq',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '-e',
    '--evaluate',
    dest='evaluate',
    action='store_true',
    help='evaluate model on validation set')
parser.add_argument(
    '--pretrained',
    dest='pretrained',
    action='store_true',
    help='use pre-trained model')
parser.add_argument(
    '--world-size',
    default=1,
    type=int,
    help='number of distributed processes')
parser.add_argument(
    '--dist-url',
    default='tcp://224.66.41.62:23456',
    type=str,
    help='url used to set up distributed training')
parser.add_argument(
    '--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--vis', action='store_true')

best_prec1 = 0

import visdom
vis = visdom.Visdom(port='7097')

visdom_transform = transforms.Compose([
        transforms.Normalize(mean = [ 0., 0., 0. ],std = [ 1/0.229, 1/0.224, 1/0.225 ]),
        transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],std = [ 1., 1., 1. ]),
                               ])

def main():
    # import pdb; pdb.set_trace()
    
    global args, best_prec1
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    # create model
    print(args.pretrained)
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'localizer_alexnet':
        model = localizer_alexnet(pretrained=args.pretrained)
    elif args.arch == 'localizer_alexnet_robust':
        model = localizer_alexnet_robust(pretrained=args.pretrained)
    print(model)

    model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    """
    github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    # TODO:
    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda()
    # torch.manual_seed(30)
    # np.random.seed(30)
    # criterion = nn.BCELoss().cuda()
    criterion = nn.MultiLabelSoftMarginLoss().cuda()
    # criterion = nn.MultiLabelMarginLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    # TODO: Write code for IMDBDataset in custom.py
    trainval_imdb = get_imdb('voc_2007_trainval')
    test_imdb = get_imdb('voc_2007_test')

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = IMDBDataset(
        trainval_imdb,
        transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        IMDBDataset(
            test_imdb,
            transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    # TODO: Create loggers for visdom and tboard
    # TODO: You can pass the logger objects to train(), make appropriate
    # modifications to train()
    writer = SummaryWriter('runs/Task_1.7')
    # if args.vis:
    


    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, writer)

        # evaluate on validation set
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            m1, m2 = validate(val_loader, model, criterion, writer, epoch)
            score = m1 * m2
            # remember best prec@1 and save checkpoint
            is_best = score > best_prec1
            best_prec1 = max(score, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)

    writer.close()



#TODO: You can add input arguments if you wish
def func_output(output):
    ksize = (output.size()[-2], output.size()[-1])
    pool = nn.MaxPool2d(kernel_size=ksize)
    sigm = nn.Sigmoid()
    return pool, sigm

def train(train_loader, model, criterion, optimizer, epoch, writer):
    # import pdb; pdb.set_trace()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.type(torch.FloatTensor).cuda(async=True)
        # target = target.type(torch.LongTensor)
        input_var = input
        target_var = target

        # TODO: Get output from model
        output = model(input_var)
        # TODO: Perform any necessary functions on the output
        # import pdb; pdb.set_trace()
        pool_operator, sigmoid_operator = func_output(output)
        pooled_output = pool_operator(output)
        imoutput = pooled_output.squeeze()
        # scaled_output = sigmoid_operator(pooled_output)
        # imoutput = scaled_output.squeeze()
        
        # TODO: Compute loss using ``criterion``
        loss = criterion(imoutput, target_var)

        # measure metrics and record loss
        m1 = metric1(imoutput.data, target_var)
        m2 = metric2(imoutput.data, target_var)
        # import pdb; pdb.set_trace()
        losses.update(loss.data[0], input.size(0))
        avg_m1.update(m1[0], input.size(0))
        avg_m2.update(m2[0], input.size(0))

        # TODO:
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        batch_num = len(train_loader)
        step = epoch*batch_num*32+(i+1)*32

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))
        writer.add_scalar("Loss_Train: ", loss.data[0], step)
        writer.add_scalar("metric1_Train: ", m1[0], step)
        writer.add_scalar("metric2_Train: ", m2[0], step)
        #TODO: Visualize things as mentioned in handout
        #TODO: Visualize at appropriate intervals

        # import pdb; pdb.set_trace()
        if epoch == 0 or epoch == 14 or epoch == 29:
            if i == 4 or i == 6:
                nums = np.random.choice(32,2,replace=False)
                gt_cls1 = np.where(target[nums[0]]==1)[0][0]
                gt_cls2 = np.where(target[nums[1]]==1)[0][0]
                heatmap1_in = torch.sigmoid(output[nums[0],gt_cls1]).squeeze()
                heatmap2_in = torch.sigmoid(output[nums[1],gt_cls2]).squeeze()
                im1 = visdom_transform(input[nums[0]])
                im2 = visdom_transform(input[nums[1]])
                vis.image(im1, opts={'title':str(epoch)+"_"+str(i)+"_im1_"+str(gt_cls1)})
                vis.heatmap(heatmap1_in, opts={'title':str(epoch)+"_"+str(i)+"_heatmap1_"+str(gt_cls1)})
                vis.image(im2, opts={'title':str(epoch)+"_"+str(i)+"_im2_"+str(gt_cls2)})
                vis.heatmap(heatmap2_in, opts={'title':str(epoch)+"_"+str(i)+"_heatmap2_"+str(gt_cls2)})
                pdb.set_trace()


        # End of train()


def validate(val_loader, model, criterion, writer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.type(torch.FloatTensor).cuda(async=True)
        input_var = input
        target_var = target

        # TODO: Get output from model
        output = model(input_var)
        # TODO: Perform any necessary functions on the output
        pool_operator, sigmoid_operator = func_output(output)
        pooled_output = pool_operator(output)
        scaled_output = sigmoid_operator(pooled_output)
        imoutput = scaled_output.squeeze()
        # TODO: Compute loss using ``criterion``
        loss = criterion(imoutput, target_var)
        
        # measure metrics and record loss
        m1 = metric1(imoutput.data, target)
        m2 = metric2(imoutput.data, target)
        # import pdb; pdb.set_trace()
        losses.update(loss.data[0], input.size(0))
        avg_m1.update(m1[0], input.size(0))
        avg_m2.update(m2[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # batch_num = len(val_loader)
        # step = epoch*batch_num*32+(i+1)*32

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      i,
                      len(val_loader),
                      batch_time=batch_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))

        #TODO: Visualize things as mentioned in handout
        #TODO: Visualize at appropriate intervals
        writer.add_scalar("Loss_Valid: ", loss.data[0], epoch)
        writer.add_scalar("metric1_Valid: ", m1[0], epoch)
        writer.add_scalar("metric2_Valid: ", m2[0], epoch)


        if epoch>20:
            nums = np.random.choice(10,3,replace=False)
            gt_cls1 = np.where(target[nums[0]]==1)[0][0]
            gt_cls2 = np.where(target[nums[1]]==1)[0][0]
            gt_cls3 = np.where(target[nums[2]]==1)[0][0]
            heatmap1_in = torch.sigmoid(output[nums[0],gt_cls1]).squeeze()
            heatmap2_in = torch.sigmoid(output[nums[1],gt_cls2]).squeeze()
            heatmap3_in = torch.sigmoid(output[nums[2],gt_cls3]).squeeze()
            im1 = visdom_transform(input[nums[0]])
            im2 = visdom_transform(input[nums[1]])
            im3 = visdom_transform(input[nums[2]])
            vis.image(im1, opts={'title':"valid"+"_"+str(i)+"_im1_"+str(gt_cls1)})
            vis.heatmap(heatmap1_in, opts={'title':"valid"+"_"+str(i)+"_heatmap1_"+str(gt_cls1)})
            vis.image(im2, opts={'title':"valid"+"_"+str(i)+"_im2_"+str(gt_cls2)})
            vis.heatmap(heatmap2_in, opts={'"valid"+title':"_"+str(i)+"_heatmap2_"+str(gt_cls2)})
            vis.image(im3, opts={'title':"valid"+"_"+str(i)+"_im2_"+str(gt_cls3)})
            vis.heatmap(heatmap2_in, opts={'"valid"+title':"_"+str(i)+"_heatmap2_"+str(gt_cls3)})



    print(' * Metric1 {avg_m1.avg:.3f} Metric2 {avg_m2.avg:.3f}'.format(
        avg_m1=avg_m1, avg_m2=avg_m2))

    return avg_m1.avg, avg_m2.avg


# TODO: You can make changes to this function if you wish (not necessary)
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def metric1(output, target):
    # TODO: Ignore for now - proceed till instructed
    from sklearn.metrics import average_precision_score
    # import pdb; pdb.set_trace()
    
    # Hyperparameter: threshold -- probability at which a predicted_class is labelled 0 or 1
    thresh = 0.5
    AP = []
    nclasses = target.shape[1]
    for cid in range(nclasses):
        pred_class = output[:,cid].detach().cpu().numpy()
        gt_class = target[:,cid].detach().cpu().numpy()

        # If class cid does not appear in gt of any of the batches
        if not np.any(gt_class):
            # then
            # If class cid is predicted (with a val>threshold) in output of any of the batches 
            if pred_class[pred_class>thresh].shape[0] > 0:
                # then assign ap=0 since it is a wrong prediction
                ap = 0
            
            # else assign ap=1 since the prediction is in line with gt
            else:
                ap = 1
        
        # else
        else:
            pred_class -= 1e-5 * gt_class
            ap = average_precision_score(gt_class, pred_class)
        
        # ap = average_precision_score(gt_class, pred_class)
        AP.append(ap)
    
    mAP = np.mean(AP)

    return [mAP]
    # return [0]


def metric2(output, target):
    #TODO: Ignore for now - proceed till instructed
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import fbeta_score
    """
    https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    https://apple.github.io/turicreate/docs/userguide/evaluation/classification.html
    https://en.wikipedia.org/wiki/F1_score

    Possible evaluation metrics for multi-object classification:
        1. Precision-Recall: --> float
            Precision - From the items that the classifier predicts as true, how many are actually true [TP/(TP+FP)]
            Recall    - From all the true items, how many does the classifier predict true              [TP/(TP+FN)]
                {
                    Using recall as a metric in this case gives 1 very quickly since most labels are 0's and that's what the network learns
                    to classify.
                    Precision = 1.0 and Recall = 1.0 within 2 epochs
                }
        2. F-scores: -- float
            F1     - Combines Precision and Recall through their Harmonic Mean [F1= HM=GM**2/AM; GM=sqrt(Precision*Recall), AM=mean(Precision,Recall)] 
            F-beta - Combines Precision and Recall through their Harmonic Mean while assigning Recall beta times the weight as Precision.
                              (1+beta**2)*Precision * Recall
                     F-beta = -------------------------------
                               (beta**2)*Precision + Recall
        3. Receiver Operating Characteristics (ROC) Curve: --> list[float], list[float]
            Computes FalsePositiveRate & TruePositiveRate
        4. Area under the Curve ROC Score: --> float
            Computes AUC ROC score
    """
    # import pdb; pdb.set_trace()    
    # criterion defines metric being used
    criterion = f1_score
    # Hyperparameter: threshold -- probability at which a predicted_class is labelled 0 or 1
    thresh = 0.5

    metric_list = []
    nclasses = target.shape[1]
    for cid in range(nclasses):
        pred_class = output[:,cid].detach().cpu().numpy()
        gt_class = target[:,cid].detach().cpu().numpy()

        # If class cid does not appear in gt of any of the batches
        if not np.any(gt_class):
            # then
            # If class cid is predicted (with a val>threshold) in output of any of the batches 
            if pred_class[pred_class>thresh].shape[0] > 0:
                # then assign ap=0 since it is a wrong prediction
                metric = 0
            
            # else assign ap=1 since the prediction is in line with gt
            else:
                metric = 1
        
        # else
        else:
            # Converting continous predicted values to binary
            binary_pred_class = np.where(pred_class>thresh, 1, 0)
            if criterion is recall_score:
                metric = criterion(gt_class, binary_pred_class)
            elif criterion is f1_score:
                metric = criterion(gt_class, binary_pred_class, average='macro')
            elif criterion is fbeta_score:
                metric = criterion(gt_class, binary_pred_class, beta=2)
            else:
                raise(RuntimeError("Specify criterion from the given 3 choices or import necessary from sklearn.metrics"))
        
        metric_list.append(metric)
        
    mean_metric = np.mean(metric_list)
    return [mean_metric]
    # return [0]


if __name__ == '__main__':
    main()