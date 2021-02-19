"""
Handles the case where we are not going to perform any sort of defence on our inputs.
"""
from defence.defence_models.base import AbstractDefence


class NoDefence(AbstractDefence):

    def __init__(self):
        super(NoDefence, self).__init__()

    def forward(self, model, images, labels):
        """
        Process the input through the model to obtain corresponding output
        :param model: The model to be used in white box setting
        :param images: The input images to be used by the model
        :param labels: Corresponding ground truth labels for the images
        :return: Model output
        """
        output = model(images)
        return output

    @property
    def is_perturbed(self):
        """
        Property to indicate if we are processing perturbed samples or not. Since `NoDefence` is not trying to use
        any perturbations, value is False. The function helps during Adversarial training to print properly formatted
        loss and accuracy metric.
        :return: False
        """
        return False

    def additional_defence_loss(self, *args, **kwargs):
        """
        Scalar value to add any auxiliary loss computed during training. Specifically used in case of AutoEncoder based
        defence in the samples.
        :return: Scalar value to be added to the loss
        """
        return 0
