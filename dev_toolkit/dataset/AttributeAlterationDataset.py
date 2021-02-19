"""
This file is going to act as an abstraction for the celebA dataset and load samples from the dataset
We are using the general pytorch framework for the purpose
"""

from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import transforms
import os

from environment_setup import PROJECT_ROOT_DIR


class AttributesDataset(Dataset):
    def __init__(self, split='train', root=os.path.join(PROJECT_ROOT_DIR, "data"), transform=None, dataset_min_val=0, dataset_max_val=1):
        """
        Generate dataset object. Please set `self.min_val` and `self.max_val` .
        It would be needed for Adversarial perturbation generation.
        :param split: (string) train/valid/test
        :param transform: Transforms to be applied on model. If None, resizes image to
                (256, 256) and scales pixels to [0, 1]
        :param dataset_max_val: maximum_value of input images
        :param dataset_min_val: minimum_value of input images
        """
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
        super(AttributesDataset, self).__init__()
        self.data = torchvision.datasets.CelebA(
            root=root,
            split=split,
            target_type='attr',
            download=True,
            transform=transform
        )
        # Range of values for the input
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
        img, label = self.data[item]
        return item, img, label

    @staticmethod
    def pred_acc(prediction, gt_label):
        """
        Each of the datasets defines their own criterion for accuracy
        :param prediction: (B, 40) tensor of logits
        :param gt_label: (B, 40) ground truth tensor
        :return: accuracy value
        """
        return (prediction >= 0).eq(gt_label).sum().item() / 40


def get_data_loader(batch_size, split, transform=None, shuffle=False, num_workers=4, dataset_min_val=0, dataset_max_val=1):
    """
    Return the dataloader object. Shared method for all train/val/test splits
    :param dataset_max_val: maximum_value of input images
    :param dataset_min_val: minimum_value of input images
    :param batch_size: int
    :param split: train/valid/test
    :param transform: Transform object. Default: None
    :param shuffle: boolean. Default: False
    :param num_workers: int. number of cores. Default 4
    :return: Dataloader object
    """
    return DataLoader(dataset=AttributesDataset(split=split, transform=transform, dataset_min_val=dataset_min_val, dataset_max_val=dataset_max_val),
                      num_workers=num_workers,
                      shuffle=shuffle,
                      batch_size=batch_size
                      )


if __name__ == '__main__':
    dataloader = get_data_loader(4, 'valid')
    dataset = dataloader.dataset
    for img_name, img, label in dataloader:
        print(img_name)
        print(img.shape)
        print(img.max())
        print(img.min())
        print(label.shape)
        # print(dataloader.dataset.pred_acc)
        break
