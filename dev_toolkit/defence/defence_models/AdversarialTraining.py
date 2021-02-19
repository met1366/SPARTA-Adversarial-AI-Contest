from defence.defence_models.base import AbstractDefence


class AdversarialTrain(AbstractDefence):

    def __init__(self, attack_method):
        super(AdversarialTrain, self).__init__()
        self.attack_method = attack_method

    def forward(self, model, images, labels):
        """
        :param model: The model being used
        :param images: Input images to the model
        :param labels: Corresponding ground truth labels
        :return: Model output
        """
        # Get a perturbed image rather than the actual one
        perturbed_image = self.attack_method.generate_perturbed_image(images, labels, train=True)
        output = model(perturbed_image)
        return output

    @property
    def is_perturbed(self):
        """
        Property to indicate if we are processing perturbed samples or not. Since `AdversarialTrain` is using
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
        return 0
