from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths
import os
import torch
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import numpy as np
from datetime import datetime

import cPickle as pkl
import network
from wsddn import WSDDN
from utils.timer import Timer

import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from datasets.factory import get_imdb
from fast_rcnn.config import cfg, cfg_from_file
import gc
import pdb

from tensorboardX import SummaryWriter
import visdom
import test

try:
    from termcolor import cprint
except ImportError:
    cprint = None


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)


# hyper-parameters
# ------------
imdb_name = 'voc_2007_trainval'
imdb_name_test = 'voc_2007_test'
cfg_file = 'experiments/cfgs/wsddn.yml'
pretrained_model = 'data/pretrained_model/alexnet_imagenet.npy'
output_dir = 'models/saved_model'
visualize = True
vis_interval = 5000

start_step = 0
end_step = 30000
lr_decay_steps = {150000}
lr_decay = 1. / 10

rand_seed = 1024
_DEBUG = False
use_tensorboard = True
use_visdom = True
log_grads = False

remove_all_log = False  # remove all historical experiments in TensorBoard
exp_name = None  # the previous experiment name in TensorBoard
# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config file and get hyperparameters
cfg_from_file(cfg_file)
lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
disp_interval = cfg.TRAIN.DISPLAY
log_interval = cfg.TRAIN.LOG_IMAGE_ITERS

# load imdb and create data later
# import pdb; pdb.set_trace()
imdb = get_imdb(imdb_name)
imdb_test = get_imdb(imdb_name_test)
rdl_roidb.prepare_roidb(imdb)
roidb = imdb.roidb
data_layer = RoIDataLayer(roidb, imdb.num_classes)

# Create network and initialize
net = WSDDN(classes=imdb.classes, debug=_DEBUG)
print(net)
network.weights_normal_init(net, dev=0.001)
if os.path.exists('pretrained_alexnet.pkl'):
    pret_net = pkl.load(open('pretrained_alexnet.pkl', 'rb'))
else:
    pret_net = model_zoo.load_url(
        'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
    pkl.dump(pret_net, open('pretrained_alexnet.pkl', 'wb'),
             pkl.HIGHEST_PROTOCOL)
own_state = net.state_dict()
for name, param in pret_net.items():
    if name not in own_state:
        continue
    if isinstance(param, Parameter):
        param = param.data
    try:
        own_state[name].copy_(param)
        print('Copied {}'.format(name))
    except:
        print('Did not find {}'.format(name))
        continue

# Move model to GPU and set train mode
net.load_state_dict(own_state)
net.cuda()
net.train()
# import pdb; pdb.set_trace()
# TODO: Create optimizer for network parameters from conv2 onwards
# (do not optimize conv1)
net_params = list(net.parameters())
optimizer = torch.optim.SGD(net_params[2:], lr=lr, momentum=momentum, weight_decay=weight_decay)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# write to tensorboardX
writer = SummaryWriter('runs/wsdnn1')
# Visdom initialization
vis = visdom.Visdom(port='8097')
# Flags for test visualizations with visdom
flag_create_plot_mAP = True
flag_create_plot_trainloss = True
# Evaluate at
eval_every = 10

# training
train_loss = 0
tp, tf, fg, bg = 0., 0., 0, 0
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()
for step in range(start_step, end_step + 1):

    # get one batch
    blobs = data_layer.forward()
    im_data = blobs['data']
    rois = blobs['rois']
    im_info = blobs['im_info']
    gt_vec = blobs['labels']
    #gt_boxes = blobs['gt_boxes']

    # forward
    # import pdb; pdb.set_trace()
    net(im_data, rois, im_info, gt_vec)
    loss = net.loss
    train_loss += loss.item()
    step_cnt += 1

    # backward pass and update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Log to screen
    if step % disp_interval == 0:
        duration = t.toc(average=False)
        fps = step_cnt / duration
        log_text = 'step %d, image: %s, loss: %.4f, fps: %.2f (%.2fs per batch), lr: %.9f, momen: %.4f, wt_dec: %.6f' % (
            step, blobs['im_name'], train_loss / step_cnt, fps, 1. / fps, lr,
            momentum, weight_decay)
        log_print(log_text, color='green', attrs=['bold'])
        re_cnt = True

    #TODO: evaluate the model every N iterations (N defined in handout)

    if step % vis_interval == 0 and step>0:
        AP = test.test_net(name='test', net=net, imdb=imdb_test, logger=writer, step=step, visualize=True, thresh = 0.001)
        # import pdb; pdb.set_trace()
        for i in range(len(AP)):
            writer.add_scalar('test_AP_class_: '+imdb._classes[i], AP[i], step)
        if flag_create_plot_mAP==True:
            vis.line(Y = np.array([np.mean(AP)]), X = np.array([step]), win="mAP_test", opts=dict(title='mAP_vs_step(test)'))
            flag_create_plot_mAP = False
        else:
            vis.line(Y = np.array([np.mean(AP)]), X = np.array([step]), win="mAP_test", update="append", opts=dict(title='mAP_vs_step(test)'))




    #TODO: Perform all visualizations here
    #You can define other interval variable if you want (this is just an
    #example)
    #The intervals for different things are defined in the handout
    # if visualize and step % vis_interval == 0:
    # print(step)
    if visualize and step % 500 == 0:
        #TODO: Create required visualizations
        if use_tensorboard:
            print('Logging to Tensorboard')
            writer.add_scalar('trainloss/', loss.data[0], step)
        if use_visdom:
            print('Logging to visdom')
            if flag_create_plot_trainloss==True:
                vis.line(Y = np.array([loss.data[0]]), X = np.array([step]), win="trainloss", opts=dict(title='train_loss'))
                flag_create_plot_trainloss = False
            else:
                vis.line(Y = np.array([loss.data[0]]), X = np.array([step]), win="trainloss", update="append", opts=dict(title='train_loss'))



    # Save model occasionally
    if (step % cfg.TRAIN.SNAPSHOT_ITERS == 0) and step > 0:
        save_name = os.path.join(
            output_dir, '{}_{}.h5'.format(cfg.TRAIN.SNAPSHOT_PREFIX, step))
        network.save_net(save_name, net)
        print('Saved model to {}'.format(save_name))

    if step in lr_decay_steps:
        lr *= lr_decay
        optimizer = torch.optim.SGD(
            net_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if re_cnt:
        tp, tf, fg, bg = 0., 0., 0, 0
        train_loss = 0
        step_cnt = 0
        t.tic()
        re_cnt = False
