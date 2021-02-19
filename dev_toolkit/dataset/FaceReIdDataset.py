"""
This file is going to act as an abstraction for the celebA dataset and load samples from the dataset
We are using the general pytorch framework for the purpose
"""
import csv

import PIL
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms
import os

from environment_setup import PROJECT_ROOT_DIR
import numpy as np


class FaceReIdDataset(Dataset):
    def __init__(self, split='train', root=os.path.join(PROJECT_ROOT_DIR, 'data'), transform=None, dataset_min_val=0,
                 dataset_max_val=1):
        """
        Generate dataset object. Please set `self.min_val` and `self.max_val` .
        It would be needed for Adversarial perturbation generation.
        :param split: split: (string) train/valid/test
        :param transform: Transformations to be applied to the image.
        :param dataset_max_val: maximum_value of input images
        :param dataset_min_val: minimum_value of input images
        """
        self.transform = transform
        super(FaceReIdDataset, self).__init__()
        if split == 'train':
            filename = 'train.csv'
        elif split == 'valid':
            filename = 'val.csv'
        elif split == 'test':
            filename = 'test.csv'
        else:
            raise AttributeError("Invalid split selection")
        csv_file = os.path.join(root, 'reid_dataset', filename)
        self.root = root
        self.base_folder = "celeba"
        self.data = self.load_data(csv_file)
        # The range of input values
        self.min_val = dataset_min_val
        self.max_val = dataset_max_val

    def __len__(self):
        """
        Total number of samples in the dataset
        :return: Integer value representing the total number of samples
        """
        return len(self.data)

    def __getitem__(self, item):
        """
        Return a single instance of the dataset object
        :param item: index from dataset
        :return: image_index, transformed image, gt_label
        """
        image, target = self.data[item]
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", image))
        if self.transform is not None:
            X = self.transform(X)
        else:
            # PyTorch does not support collate operation on jpeg data. Hence, we need to convert it to Tensor
            X = transforms.ToTensor()(X)
        return item, X, target

    def load_data(self, csv_file):
        """
        The function reads data stored in the form of a csv file
        :param csv_file: filename obtianed based on the split
        :return: list of tuples of image, label pair
        """
        image, label = [], []
        with open(csv_file) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                image.append(row['image'])
                label.append(int(row['label']))
        return list(zip(image, label))

    @staticmethod
    def pred_acc(prediction, gt_label):
        """
        Each of the datasets defines their own criterion for accuracy
        :param prediction: (B, C) tensor of logits
        :param gt_label: (B, 1) ground truth tensor
        :return: The accuracy value
        """
        _, predicted_label = torch.max(prediction, 1)
        return predicted_label.eq(gt_label).sum().item()


def get_reid_data_loader(batch_size, split, transform=None, shuffle=False, num_workers=4, use_mtcnn=True,
                         dataset_min_val=0, dataset_max_val=1):
    """
    Return the dataloader object. Shared method for all train/val/test splits
    :param dataset_max_val: maximum_value of input images
    :param dataset_min_val: minimum_value of input images
    :param use_mtcnn: To use images cropped based on MTCNN
    :param batch_size: int
    :param split: train/valid/test
    :param transform: Transform object. Default: None
    :param shuffle: boolean. Default: False
    :param num_workers: int. number of cores. Default 4
    :return: Dataloader object
    """
    if use_mtcnn:
        from dataset import MTCNNFaceReIdDataset
        dataset = MTCNNFaceReIdDataset(split=split, transform=transform, dataset_min_val=dataset_min_val,
                                       dataset_max_val=dataset_max_val)
    else:
        dataset = FaceReIdDataset(split=split, transform=transform, dataset_min_val=dataset_min_val,
                                  dataset_max_val=dataset_max_val)
    return DataLoader(dataset=dataset,
                      num_workers=num_workers,
                      shuffle=shuffle,
                      batch_size=batch_size
                      )


"""
Series of utility methods that may help in visualizing and understanding the dataset statistics better.
"""


def get_groups():
    attr_arr = np.asarray(['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
                           'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry',
                           'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
                           'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
                           'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face',
                           'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
                           'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
                           'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'])
    # location dictionary
    attr_dict = {}
    for idx, name in enumerate(attr_arr):
        attr_dict[idx] = name
    return attr_dict


stats = np.zeros((40, 1))


def update_val(label_tensor):
    global stats
    label_tensor_numpy = label_tensor.cpu().numpy().reshape(40, 1)
    # list of 40 values
    stats = stats + label_tensor_numpy


def pretty_print(stats):
    attr_dict = get_groups()
    for idx, counts in enumerate(stats):
        print(f"{idx} has {attr_dict[idx]}:     {counts}")


if __name__ == '__main__':
    dataloader = get_reid_data_loader(batch_size=128, split='test')
    dataset = dataloader.dataset
    for _, img, label in dataloader:
        print(label)
        print(img.shape)
        break
