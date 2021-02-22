import torch

from defence.defence_models.base import AbstractDefence
from networks import DummyNetwork


class AutoEncoderAdversarialDefence(AbstractDefence):

    def __init__(self, attack_method, model):
        super(AutoEncoderAdversarialDefence, self).__init__()
        self.attack_method = attack_method
        self.model = model

    def forward(self, model, images, labels):
        """
        :param model: The model being passed
        :param images: Images
        :param labels: Corresponding labels
        :return: perturbed sample
        """
        # generate perturbed samples for feeding into the model
        noisy_input = self.attack_method.generate_perturbed_image(images, labels, train=True)
        output = model(noisy_input, images)
        return output

    @property
    def is_perturbed(self):
        """
        Property to indicate if we are processing perturbed samples or not. Since `AutoEncoderAdversarialDefence` is using
        perturbations, value is True. The function helps during Adversarial training to print properly formatted
        loss and accuracy metric.
        :return: True
        """
        return True

    def additional_defence_loss(self, *args, **kwargs):
        """
        Scalar value to add any auxiliary loss computed during training. Specifically used in case of AutoEncoder based
        defence in the samples.
        :return: Scalar value to be added to the loss
        """
        return self.model.additional_loss


if __name__ == '__main__':
    defence_network = AutoEncoderAdversarialDefence()
    model = DummyNetwork(name='dummy')
    x = torch.randn((10, 3, 256, 256))  # B, num_channel, H, W
    output = defence_network(model, x, None)
    print(output.shape)
    print(defence_network.additional_defence_loss())
