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

from sklearn.neighbors import NearestNeighbors
import pickle

# from q3_pytorch_caffe import CaffeNet
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
        pool_feat = self.pool(x)
        flat_x = pool_feat.view(N, -1)
        out = self.fc1(flat_x)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out, pool_feat

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
    # model = models.resnet18(pretrained=False)
    model = CaffeNet(num_classes=len(VOCDataset.CLASS_NAMES), inp_size=227, c_dim=3).to(device)
    # model.fc = nn.Linear(512,20)
    model.load_state_dict(torch.load('/home/biorobotics/VLR/Visual-Learning-Recognition-16824/HW1/q2_caffe_models_v1/model_at_epoch_40.pth'))
    model = model.to(device)
    model.eval()
    test_loader, index_list = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test')

    # print(len(index_list))

    # img_idx = np.random.choice(len(test_loader),size=3,replace=False)
    # _, index_list = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test')
    
    index_list = np.array(index_list)
    # import ipdb; ipdb.set_trace()
    img1_idx = np.where(index_list=='000002')[0][0] # 000002.jpg  train
    img2_idx = np.where(index_list=='000029')[0][0] # 000029.jpg # dog
    img3_idx = np.where(index_list=='000135')[0][0] # 000135.jpg # car
    # img1_idx = 1
    # print(img1_idx)

    
    response_list = []
    
    # for batch_idx, (data, target, wgt) in enumerate(test_loader):
    #     print(batch_idx)
    #     data = data.to(device)
    #     # model.avgpool.register_forward_hook(get_layer_response('avgpool'))
    #     out, layer_response = model(data)
    #     response_list.append(layer_response.detach().cpu().numpy())

    # print(len(response_list))
    # with open("/home/biorobotics/VLR/Visual-Learning-Recognition-16824/HW1/caffe_feat.txt", "wb") as file:
    #     pickle.dump(response_list, file)


    with open("/home/biorobotics/VLR/Visual-Learning-Recognition-16824/HW1/caffe_feat.txt", "rb") as file: 
        response_list = pickle.load(file)
    print(response_list[0].reshape(-1))
    response_list_2d = []
    for i in range(len(response_list)):
        # print(i)
        # response_list_2d.append(response_list[i].view(-1).detach().cpu().numpy())
        response_list_2d.append(response_list[i].reshape(-1))

    print(np.shape(response_list_2d))
    neigh = NearestNeighbors(n_neighbors=4)
    neigh.fit(np.array(response_list_2d))

    # import ipdb; ipdb.set_trace()
    closest_to_img1 = neigh.kneighbors(np.array(response_list_2d[img1_idx]).reshape(1,-1))
    closest_to_img2 = neigh.kneighbors(np.array(response_list_2d[img2_idx]).reshape(1,-1))
    closest_to_img3 = neigh.kneighbors(np.array(response_list_2d[img3_idx]).reshape(1,-1))

    print("Input Image Num: ", index_list[img1_idx], " Output image nums: ", index_list[closest_to_img1[1][0,0]], index_list[closest_to_img1[1][0,1]], index_list[closest_to_img1[1][0,2]], index_list[closest_to_img1[1][0,3]])
    print("Input Image Num: ", index_list[img2_idx], " Output image nums: ", index_list[closest_to_img2[1][0,0]], index_list[closest_to_img2[1][0,1]], index_list[closest_to_img2[1][0,2]], index_list[closest_to_img2[1][0,3]])
    print("Input Image Num: ", index_list[img3_idx], " Output image nums: ", index_list[closest_to_img3[1][0,0]], index_list[closest_to_img3[1][0,1]], index_list[closest_to_img3[1][0,2]], index_list[closest_to_img3[1][0,3]])

    
    # print(closest_to_img1)
    # VOCDataset.get_class_name(closest_to_img1[0])

if __name__ == "__main__":
    args, device = utils.parse_args()
    main()