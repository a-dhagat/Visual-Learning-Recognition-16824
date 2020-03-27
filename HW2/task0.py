from __future__ import print_function
import _init_paths
from datasets.factory import get_imdb
import visdom
import numpy as np
import cv2

imdb = get_imdb('voc_2007_trainval')
# print(imdb)

img_path_at_2020 = imdb.image_path_at(2020)
# print(img_path_at_2020)

img_name = img_path_at_2020.split('/')[-1]
img_idx = img_name.split('.')[0]
annotation = imdb._load_pascal_annotation(img_idx)
# print(annotation)
gt_bounding_box = annotation['boxes']
gt_roidb = imdb.gt_roidb()
roidb = imdb._load_selective_search_roidb(gt_roidb)
roi2020 = roidb[2020]

vis = visdom.Visdom(port='8097')
img_2020 = cv2.imread(img_path_at_2020)
for box in gt_bounding_box:
    bbox = tuple(box)
    cv2.rectangle(img_2020, bbox[0:2], bbox[2:4], (0,0,255), 2)
# for box in roi2020['boxes'][0:10]:
#     bbox = tuple(box)
#     cv2.rectangle(img_2020, bbox[0:2], bbox[2:4], (122,0,0), 2)

img_2020_new = img_2020[:,:,-1::-1]
img_2020_new = np.transpose(img_2020_new,(2,0,1))
vis.image(img_2020_new)