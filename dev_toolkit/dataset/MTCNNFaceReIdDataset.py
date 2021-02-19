import os

import PIL
import torch

from dataset import FaceReIdDataset
from environment_setup import PROJECT_ROOT_DIR
from networks.utils.mtcnn import MTCNN


class MTCNNFaceReIdDataset(FaceReIdDataset):
    def __init__(self, split='train', root=os.path.join(PROJECT_ROOT_DIR, 'data'), transform=None, dataset_min_val=-1, dataset_max_val=1):
        super(MTCNNFaceReIdDataset, self).__init__(split=split, root=root, transform=transform)
        self.mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        # The range of input values
        self.min_val = dataset_min_val
        self.max_val = dataset_max_val

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
        X, prob = self.mtcnn(X, return_prob=True)
        if prob is None:
            print(f"culprit image is {image}")
        return item, X, target