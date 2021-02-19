"""
This code is obtained almost entirely from
https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py
"""


import os

import torch
from torch import nn
from torch.nn import functional as F

from environment_setup import PROJECT_ROOT_DIR
from networks import Network
from networks.utils.inception_resnet_utils import Block35, Mixed_6a, Block17, Mixed_7a, Block8, BasicConv2d


class InceptionResnetV1(nn.Module, Network):
    """Inception Resnet V1 model with optional loading of pretrained weights.
    Model parameters can be loaded based on pretraining on the VGGFace2 or CASIA-Webface
    datasets. Pretrained state_dicts are automatically downloaded on model instantiation if
    requested and cached in the torch cache. Subsequent instantiations use the cache rather than
    redownloading.
    Keyword Arguments:
        pretrained {str} -- Optional pretraining dataset. Either 'vggface2' or 'casia-webface'.
            (default: {None})
        classify {bool} -- Whether the model should output classification probabilities or feature
            embeddings. (default: {False})
        num_classes {int} -- Number of output classes. If 'pretrained' is set and num_classes not
            equal to that used for the pretrained model, the final linear layer will be randomly
            initialized. (default: {None})
        dropout_prob {float} -- Dropout probability. (default: {0.6})
    """
    def __init__(self, name, pretrained, classify, num_classes, dropout_prob, device, pre_train_wt_folder):
        super().__init__()
        super(nn.Module, self).__init__(name=name)
        # Set simple attributes
        self.pretrained = pretrained
        self.classify = classify
        self.num_classes = num_classes
        self.pre_train_wt_folder = pre_train_wt_folder

        if pretrained == 'vggface2':
            tmp_classes = 8631
        elif pretrained == 'casia-webface':
            tmp_classes = 10575
        elif pretrained is None and self.classify and self.num_classes is None:
            raise Exception('If "pretrained" is not specified and "classify" is True, "num_classes" must be specified')

        # Define layers
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2)
        self.repeat_1 = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
        )
        self.mixed_6a = Mixed_6a()
        self.repeat_2 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
        )
        self.mixed_7a = Mixed_7a()
        self.repeat_3 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
        )
        self.block8 = Block8(noReLU=True)
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.last_linear = nn.Linear(1792, 512, bias=False)
        self.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)

        if pretrained is not None:
            self.logits = nn.Linear(512, tmp_classes)
            self.load_weights(self, pretrained)

        if self.classify and self.num_classes is not None:
            self.logits = nn.Linear(512, self.num_classes)

        if device is not None:
            self.device = device
            self.to(device)

    def forward(self, x):
        """Calculate embeddings or logits given a batch of input image tensors.
        Arguments:
            x {torch.tensor} -- Batch of image tensors representing faces.
        Returns:
            torch.tensor -- Batch of embedding vectors or multinomial logits.
        """
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = self.last_linear(x.view(x.shape[0], -1))
        x = self.last_bn(x)
        if self.classify:
            x = self.logits(x)
        else:
            x = F.normalize(x, p=2, dim=1)
        return x

    def load_weights(self, mdl, name):
        """Download pretrained state_dict and load into model.
        Arguments:
            mdl {torch.nn.Module} -- Pytorch model.
            name {str} -- Name of dataset that was used to generate pretrained state_dict.
        Raises:
            ValueError: If 'pretrained' not equal to 'vggface2' or 'casia-webface'.
        """
        if name == 'vggface2':
            path = 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt'
        elif name == 'casia-webface':
            path = 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180408-102900-casia-webface.pt'
        else:
            raise ValueError('Pretrained models only exist for "vggface2" and "casia-webface"')

        model_dir = os.path.join(PROJECT_ROOT_DIR, self.pre_train_wt_folder)
        os.makedirs(model_dir, exist_ok=True)
        model_name = os.path.join(model_dir, name + '.pt')
        if not os.listdir(model_dir):
            # Directory is empty so we are loading the model for the first time
            torch.hub.download_url_to_file(url=path, dst=model_name)
        else:
            print(f"Using the cached weights from {model_name}")
        state_dict = torch.load(model_name)
        mdl.load_state_dict(state_dict)
        # Freeze these parameters
        for param in mdl.parameters():
            param.requires_grad = False

    def save(self, model_number, checkpoint_dir):
        """
        Utility function to save the model
        :param index: The counter added to model name to distinguish different models
        :param checkpoint_dir: Location where to save the model
        :return: None
        """
        # Store the model snapshot
        filename = self.name + '{:d}'.format(model_number) + '.pth'
        filename = os.path.join(checkpoint_dir, filename)
        torch.save(self.state_dict(), filename)
        print('Saving snapshot to {:s}'.format(filename))

    def load(self, model_number, checkpoint_dir):
        """
        Utility function to load the model
        :param model_number: Specific counter added to model name to distinguish different models
        :param checkpoint_dir: Location from where to load the model
        :return: None
        """
        filename = self.name + '{:d}'.format(model_number) + '.pth'
        s_file = os.path.join(checkpoint_dir, filename)
        print('Restoring mode snapshots from {:s}'.format(filename))
        self.load_state_dict(torch.load(s_file))
        print('Restored')
