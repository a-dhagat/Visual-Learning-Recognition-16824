# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import imageio
import numpy as np
import os
import xml.etree.ElementTree as ET

import torch
import torch.nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class VOCDataset(Dataset):
    CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                   'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                   'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    INV_CLASS = {}
    for i in range(len(CLASS_NAMES)):
        INV_CLASS[CLASS_NAMES[i]] = i

    def __init__(self, split, size, data_dir='VOCdevkit/VOC2007/'):
        super().__init__()
        self.split = split
        self.data_dir = data_dir
        self.size = size
        self.img_dir = os.path.join(data_dir, 'JPEGImages')
        self.ann_dir = os.path.join(data_dir, 'Annotations')

        split_file = os.path.join(data_dir, 'ImageSets/Main', split + '.txt')
        with open(split_file) as fp:
            self.index_list = [line.strip() for line in fp]

        self.anno_list = self.preload_anno()
        if split == 'train' or 'trainval':
            self.img_transforms = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.RandomCrop(227),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
                ]
                )
        if split == 'test':
            self.img_transforms = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(227),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
                ]
                )

    @classmethod
    def get_class_name(cls, index):
        return cls.CLASS_NAMES[index]

    @classmethod
    def get_class_index(cls, name):
        return cls.INV_CLASS[name]

    def __len__(self):
        return len(self.index_list)

    # def preload_anno(self):
    #     """
    #     :return: a list of lables. each element is in the form of [class, weight],
    #      where both class and weight are a numpy array in shape of [20],
    #     """
    #     object_less_images = []
    #     label_list = []
    #     # import ipdb; ipdb.set_trace()
    #     for index in self.index_list:
    #         fpath = os.path.join(self.ann_dir, index + '.xml')
    #         tree = ET.parse(fpath)
    #         # TODO: insert your code here, preload labels
    #         label = np.zeros(20)
    #         weight = np.ones(20)
    #         root = tree.getroot()
    #         for attr_idx in range(len(root)):
    #             if root[attr_idx].tag == 'object':
    #                 # class_idx = INV_CLASS[root[i][0].text]
    #                 class_idx = self.get_class_index(root[attr_idx][0].text)
    #                 label[class_idx] = 1.0
    #                 difficult = (root[attr_idx][-2].text)
    #                 if (difficult) == '1.0':
    #                     weight[class_idx] = 0.0
    #         if not(label.any()==1.0):
    #             print("Catching!")
    #             object_less_images.append(index)

    #         label_list.append([label, weight])

    #     if(object_less_images != []):
    #         print(object_less_images)
    #         quit()
    #     return label_list

    def preload_anno(self):
        """
        :return: a list of lables. each element is in the form of [class, weight],
         where both class and weight are a numpy array in shape of [20],
        """
        object_less_images = []
        label_list = []
        # import ipdb; ipdb.set_trace()
        for index in self.index_list:
            fpath = os.path.join(self.ann_dir, index + '.xml')
            tree = ET.parse(fpath)
            # TODO: insert your code here, preload labels
            label = np.zeros(20)
            weight = np.ones(20)
            root = tree.getroot()

            objects = []
            diff = []
            for attr_idx in range(len(root)):
                if root[attr_idx].tag == 'object':
                    obj_name = root[attr_idx][0].text
                    difficult = (root[attr_idx][-2].text)
                    objects.append(obj_name)
                    diff.append(difficult)

            object_set = set(objects)
            objects = np.array(objects)
            diff = np.array(diff)

            for obj in object_set:
                idxs = np.where(objects==obj)[0]
                obj_diff = diff[idxs]
                obj_diff = np.where(obj_diff=='1.0', True, False)

                class_idx = self.get_class_index(obj)
                label[class_idx] = 1.0

                if obj_diff.all() == True:
                    weight[class_idx] = 0.0
            
            if not(label.any()==1.0):
                print("Catching!")
                object_less_images.append(index)

            label_list.append([label, weight])

        if(object_less_images != []):
            print(object_less_images)
            quit()
        return label_list

    def __getitem__(self, index):
        """
        :param index: a int generated by Dataloader in range [0, __len__()]
        :return: index-th element
        image: FloatTensor in shape of (C, H, W) in scale [-1, 1].
        label: LongTensor in shape of (Nc, ) binary label
        weight: FloatTensor in shape of (Nc, ) difficult or not.
        """
        # ipdb.set_trace()
        findex = self.index_list[index]
        # print("\nIndex:\n", index)
        # print("findex:\n", int(findex))
        # print(len(self.anno_list))
        # print(len(self.index_list))
        # quit()
        fpath = os.path.join(self.img_dir, findex + '.jpg')
        # TODO: insert your code here. hint: read image, find the labels and weight.
        img = Image.open(fpath)
        lab_vec = self.anno_list[index][0]
        wgt_vec = self.anno_list[index][1]

        # Canonical image resizing
        # new_size = (256,256)
        # img = img.resize(new_size)
        # img = np.array(img)
        # img = np.transpose(img, (2,0,1))
        # print(type(img))

        image = torch.FloatTensor(self.img_transforms(img))
        # image = torch.FloatTensor((img))
        label = torch.FloatTensor(lab_vec)
        wgt = torch.FloatTensor(wgt_vec)
        return image, label, wgt

