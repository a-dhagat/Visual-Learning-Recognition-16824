import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
model_urls = {
        'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(imdb):
    #TODO: classes: list of classes
    #TODO: class_to_idx: dictionary with keys=classes and values=class index
    #If you did Task 0, you should know how to set these values from the imdb
    classes = imdb._classes
    class_to_idx = imdb._class_to_ind
    return classes, class_to_idx


def make_dataset(imdb, class_to_idx):
    #TODO: return list of (image path, list(+ve class indices)) tuples
    #You will be using this in IMDBDataset
    dataset_list = []
    image_index = imdb._load_image_set_index()
    # img_path = []
    # class_indices = []
    for idx in image_index:
        img_path = imdb.image_path_from_index(idx)
        annotations = imdb._load_pascal_annotation(idx)
        class_indices = annotations['gt_classes']
        # class_indices.append(annotation['gt_classes'])
        dataset_list.append((img_path, class_indices))

    return dataset_list


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)



class LocalizerAlexNet(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNet, self).__init__()
        #TODO: Define model
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=(11,11), stride=(4,4), padding=(2,2))
        # self.relu1 = lambda x: F.relu(x, inplace=True)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), dilation=(1,1), ceil_mode=False)
        # self.conv2 = nn.Conv2d(64, 192, kernel_size=(5,5), stride=(1,1), padding=(2,2))
        # self.relu2 = lambda x: F.relu(x, inplace=True)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), dilation=(1,1), ceil_mode=False)
        # self.conv3 = nn.Conv2d(192, 384, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        # self.relu3 = lambda x: F.relu(x, inplace=True)
        # self.conv4 = nn.Conv2d(384, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        # self.relu4 = lambda x: F.relu(x, inplace=True)
        # self.conv5 = nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        # self.relu5 = lambda x: F.relu(x, inplace=True)

        # self.conv6 = nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1))
        # self.relu6 = lambda x: F.relu(x, inplace=True)
        # self.conv7 = nn.Conv2d(256, 256, kernel_size=(1,1), stride=(1,1))
        # self.relu7 = lambda x: F.relu(x, inplace=True)
        # self.conv8 = nn.Conv2d(256, 20, kernel_size=(1,1), stride=(1,1))

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11,11), stride=(4,4), padding=(2,2)),
            nn.ReLU(inplace=True),
            # lambda x: F.relu(x, inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), dilation=(1,1), ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=(5,5), stride=(1,1), padding=(2,2)),
            nn.ReLU(inplace=True),
            # lambda x: F.relu(x, inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), dilation=(1,1), ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            # lambda x: F.relu(x, inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            # lambda x: F.relu(x, inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            # lambda x: F.relu(x, inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1)),
            nn.ReLU(inplace=True),
            # lambda x: F.relu(x, inplace=True),
            nn.Conv2d(256, 256, kernel_size=(1,1), stride=(1,1)),
            nn.ReLU(inplace=True),
            # lambda x: F.relu(x, inplace=True),
            nn.Conv2d(256, 20, kernel_size=(1,1), stride=(1,1)),
        )


    def forward(self, x):
        #TODO: Define forward pass
        x = self.features(x)
        x = self.classifier(x)
        return x



class LocalizerAlexNetHighres(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNetHighres, self).__init__()
        #TODO: Ignore for now until instructed









    def forward(self, x):
        #TODO: Ignore for now until instructed










        return x



def localizer_alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    import torchvision
    from torch.nn.parameter import Parameter

    model = LocalizerAlexNet(**kwargs)
    #TODO: Initialize weights correctly based on whethet it is pretrained or
    # not
    print(pretrained)
    if pretrained:
        print("Pretrain true\n")
        alexnet = torchvision.models.alexnet(pretrained=True)
        alexnet_pretrained = alexnet.state_dict()
        
        """
        https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113
        """
        print("Loading features:\n")
        own_state = model.state_dict()
        for name, param in alexnet_pretrained.items():
            if name not in own_state:
                # print("Not in own state ", name)
                continue
            if isinstance(param, Parameter):
                param = param.data
        
            # Only need to retain/copy pretrained weights of features sequential, not classifier sequential
            if 'features' in name:
                own_state[name].copy_(param)
                print(name)

    else:
        """
        https://discuss.pytorch.org/t/how-to-fix-define-the-initialization-weights-seed/20156
        """

        for feature_layer in model.features:
            if isinstance(feature_layer, nn.Conv2d):
                nn.init.xavier_normal_(feature_layer.weight.data)
    
    for classif_layer in model.classifier:
        if isinstance(classif_layer, nn.Conv2d):
            print(classif_layer)
            nn.init.xavier_normal_(classif_layer.weight.data)
    
    return model
    



def localizer_alexnet_robust(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNetRobust(**kwargs)
    #TODO: Ignore for now until instructed








    return model




class IMDBDataset(data.Dataset):
    """A dataloader that reads imagesfrom imdbs
    Args:
        imdb (object): IMDB from fast-rcnn repository
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, list(+ve class indices)) tuples
    """
    import numpy as np

    def __init__(self, imdb, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(imdb)
        imgs = make_dataset(imdb, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images, what's going on?"))
        self.imdb = imdb
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a binary vector with 1s
                                   for +ve classes and 0s for -ve classes
                                   (it can be a numpy array)
        """
        # TODO: Write this function, look at the imagenet code for inspiration
        
        # Extract image path and ground_truth classes from dataset_list (imgs)
        img_path = self.imgs[index][0]
        gt_classes = self.imgs[index][1]

        # Open image using PIL at img_path
        img = default_loader(img_path)
        # img = Image.open(img_path)
        # img = self.transform(img)
        
        # Create a binary vector of classes for img at index
        import torch
        target = torch.from_numpy(np.zeros(len(self.classes)))
        for i in gt_classes:
            if i < len(self.classes):
                target[i] = 1.0
        # target = target.astype(np.uint8)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
