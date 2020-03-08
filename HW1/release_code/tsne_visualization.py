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

from sklearn.manifold import TSNE
import pickle

args, device = utils.parse_args()

test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test')


response_list = []
# with open("/home/biorobotics/VLR/Visual-Learning-Recognition-16824/HW1/resnet_feat.txt", "rb") as file: 
#         response_list = pickle.load(file)
    # print(response_list[0].view(-1))

with open("/home/biorobotics/VLR/Visual-Learning-Recognition-16824/HW1/caffe_feat.txt", "rb") as file: 
        response_list = pickle.load(file)

targets = []
import numpy as np
for batch_idx, (data, target, wgt) in enumerate(test_loader):
    print(batch_idx)
    # data = data.to(device)
    # model.avgpool.register_forward_hook(get_layer_response('avgpool'))
    # out = model(data)
    # response_list.append(activation['avgpool'])
    targets.append(target.detach().cpu().numpy().astype(np.int))

response_list_2d = []
for i in range(len(response_list)):
    # print(i)
    # response_list_2d.append(response_list[i].view(-1).detach().cpu().numpy())
    response_list_2d.append(response_list[i].reshape(-1))

x = TSNE(n_components=2).fit_transform(np.array(response_list_2d))

t = np.transpose(targets,(1,2,0))
t1 = np.argmax(t,axis=1)
t2 = t1[:,0]

CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                   'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                   'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

import seaborn as sns
palette = np.array(sns.color_palette("hls", 20))

y_dict = {}
for i in range(20):
    y_dict[str(palette[i])] = CLASS_NAMES[i]


import matplotlib.pyplot as plt

# import ipdb; ipdb.set_trace()
fig = plt.figure(figsize=(10,10),dpi=80)
# for i in range(1000):
#     plt.scatter(x[i,0], x[i,1], lw=0, s=40, c=palette[t2[i]], label=y_dict[str(palette[t2[i]])])

plt.scatter(x[:1000,0], x[:1000,1], lw=0, s=40, c=palette[t2[:1000]] ) #, label=y_dict[palette[t2[:1000]]])

for i in range(20):
    plt.scatter([], [], c=palette[i], label=y_dict[str(palette[i])] )
plt.legend()
plt.show()