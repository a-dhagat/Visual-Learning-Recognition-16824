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

activation = {}
def get_layer_response(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def main():
    # model = models.resnet18(pretrained=False)
    # model.fc = nn.Linear(512,20)
    # model.load_state_dict(torch.load('/home/biorobotics/VLR/Visual-Learning-Recognition-16824/HW1/q4_resnet18_models_v1/model_at_epoch_9.pth'))
    # model = model.to(device)
    # model.eval()
    # test_loader, index_list = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test')

    # print(len(index_list))

    # img_idx = np.random.choice(len(test_loader),size=3,replace=False)
    _, index_list = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test')
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
    #     model.avgpool.register_forward_hook(get_layer_response('avgpool'))
    #     out = model(data)
    #     response_list.append(activation['avgpool'])

    # print(len(response_list))
    # with open("/home/biorobotics/VLR/Visual-Learning-Recognition-16824/HW1/resnet_feat.txt", "wb") as file:
    #     pickle.dump(response_list, file)


    with open("/home/biorobotics/VLR/Visual-Learning-Recognition-16824/HW1/resnet_feat.txt", "rb") as file: 
        response_list = pickle.load(file)
    # print(response_list[0].view(-1))
    response_list_2d = []
    for i in range(len(response_list)):
        # print(i)
        response_list_2d.append(response_list[i].view(-1).detach().cpu().numpy())

    # print(np.shape(response_list_2d))
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