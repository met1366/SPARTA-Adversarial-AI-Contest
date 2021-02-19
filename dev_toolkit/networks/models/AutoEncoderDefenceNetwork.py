"""
This network shall act as our base class and would allow us to test the basic implementation.
It will be trained from scratch and so would be useful when we try to perform the adversarial
training portion.
"""

import torch.nn as nn
import torch
import os
import torch.nn.functional as F

from networks.models.AbstractNetwork import Network


def create_conv_block(in_channel, out_channel, kernel_size, stride=2, dropout_p=0):
    """
    Convolution Block
    :param stride: stride applied to the conv kernel
    :param in_channel: Number of input channels to the conv kernel
    :param out_channel: Number of output channels of conv kernel
    :param kernel_size: size of the kernel applied
    :param dropout_p: dropout probability used
    :return: Sequential object containing Conv->ReLU->Dropout
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride),
        nn.ReLU(),
        nn.Dropout(dropout_p)
    )


def create_trasnpose_conv_block(in_channel, out_channel, kernel_size, stride=2, dropout_p=0):
    """
    Transpose Convolution Block
    :param in_channel: Number of input channels to the TransposeConv kernel
    :param out_channel: Number of output channels of TransposeConv kernel
    :param kernel_size: size of the kernel applied
    :param stride: stride applied to the TransposeConv kernel
    :param dropout_p: dropout probability used
    :return: Sequential object containing TransposeConv->ReLU->Dropout
    """
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride),
        nn.ReLU(),
        nn.Dropout(dropout_p)
    )


class AutoEncoderDefenceNetwork(nn.Module, Network):
    def __init__(self, name, num_classes=2):
        super(AutoEncoderDefenceNetwork, self).__init__()
        super(nn.Module, self).__init__(name=name)
        self.num_classes = num_classes
        self.additional_loss = 0
        aut_enc_down1 = create_conv_block(in_channel=3, out_channel=64, kernel_size=3, stride=2)
        aut_enc_down2 = create_conv_block(in_channel=64, out_channel=128, kernel_size=3, stride=2)
        aut_enc_down3 = create_conv_block(in_channel=128, out_channel=256, kernel_size=3, stride=2)
        aut_enc_down4 = create_conv_block(in_channel=256, out_channel=512, kernel_size=3, stride=2)
        # Upsampling layers
        aut_enc_up1 = create_trasnpose_conv_block(in_channel=512, out_channel=256, kernel_size=3, stride=2)
        aut_enc_up2 = create_trasnpose_conv_block(in_channel=256, out_channel=128, kernel_size=3, stride=2)
        aut_enc_up3 = create_trasnpose_conv_block(in_channel=128, out_channel=64, kernel_size=3, stride=2)
        aut_enc_up4 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=2, output_padding=1)
        self.enc_dec = nn.Sequential(
            aut_enc_down1, aut_enc_down2, aut_enc_down3, aut_enc_down4,
            aut_enc_up1, aut_enc_up2, aut_enc_up3, aut_enc_up4
        )
        # Normal Conv layers in order to get the final classification
        self.ConvLayer1 = create_conv_block(in_channel=3, out_channel=64, kernel_size=3, stride=2)
        self.ConvLayer2 = create_conv_block(in_channel=64, out_channel=128, kernel_size=3, stride=2)
        self.ConvLayer3 = create_conv_block(in_channel=128, out_channel=256, kernel_size=3, stride=2)
        self.ConvLayer4 = create_conv_block(in_channel=256, out_channel=512, kernel_size=3, stride=2)
        self.Linear1 = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=15)

    def forward(self, noisy_images, actual_images=None):
        # Pass through the Denoising Autoencoder setting first
        x_hat = self.enc_dec(noisy_images)
        # In training time, we have the original images present which we can denoise and learn upon
        if actual_images is not None:
            self.additional_loss = F.mse_loss(x_hat, actual_images)
        else:
            self.additional_loss = 0
        x_hat = self.ConvLayer1(x_hat)
        x_hat = self.ConvLayer2(x_hat)
        x_hat = self.ConvLayer3(x_hat)
        x_hat = self.ConvLayer4(x_hat)
        x_hat = self.Linear1(x_hat)
        x_hat = x_hat.view(x_hat.shape[0], -1)  # from B, C, 1, 1 -> B, C
        return x_hat

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
    network = AutoEncoderDefenceNetwork(name="sample")
    x = torch.randn((10, 3, 256, 256))  # B, num_channel, H, W
    y = torch.randn((10, 3, 256, 256))  # B, num_channel, H, W
    output = network(x, y)
    print(output.shape)
    print(network.additional_loss)
