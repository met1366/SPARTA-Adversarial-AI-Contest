"""
This network shall act as our base class and would allow us to test the basic implementation.
It will be trained from scractch and so would be useful when we try to perform the adversarial
training portion.
"""

import torch.nn as nn
import torch
import os

from networks.models.AbstractNetwork import Network


def create_conv_block(in_channel, out_channel, kernel_size, stride=1, dropout_p=0):
    """
    :param stride: Stride used for the convolution operation
    :param in_channel: Number of input channels to the conv kernel
    :param out_channel: Number of output channels of conv kernel
    :param kernel_size: size of the kernel applied
    :param dropout_p: dropout probability used
    :return: Sequential object containing Conv->MaxPool->ReLU->Dropout
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride),
        nn.MaxPool2d(kernel_size=2),
        nn.ReLU(),
        nn.Dropout(dropout_p)
    )


class DummyNetwork(nn.Module, Network):
    def __init__(self, name, num_classes=40):
        super(DummyNetwork, self).__init__()
        super(nn.Module, self).__init__(name=name)
        self.num_classes = num_classes
        self.ConvLayer1 = create_conv_block(in_channel=3, out_channel=64, kernel_size=3)
        self.ConvLayer2 = create_conv_block(in_channel=64, out_channel=128, kernel_size=3)
        self.ConvLayer3 = create_conv_block(in_channel=128, out_channel=256, kernel_size=3)
        self.ConvLayer4 = create_conv_block(in_channel=256, out_channel=512, kernel_size=3)
        self.Linear1 = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=14)

    def forward(self, x):
        x = self.ConvLayer1(x)
        x = self.ConvLayer2(x)
        x = self.ConvLayer3(x)
        x = self.ConvLayer4(x)
        x = self.Linear1(x)
        x = x.view(x.shape[0], -1)  # from B, 40, 1, 1 -> B, 40
        return x

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


if __name__ == '__main__':
    network = DummyNetwork(name="sample")
    x = torch.randn((10, 3, 256, 256))  # B, num_channel, H, W
    output = network(x)
    print(output.shape)
    print(network.additional_loss)
