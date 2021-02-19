import torch
from attacks.attack_models.AbstractAttack import AbstractAttack


class PGD(AbstractAttack):
    def __init__(self, model, save_folder, loss_fn, epsilon=0.3, alpha=0.01, num_iterations=40, targeted=False,
                 min_val=0, max_val=1):
        super(PGD, self).__init__(model=model, save_folder=save_folder, min_val=min_val, max_val=max_val)
        self.epsilon = epsilon
        self.alpha = alpha
        self.set_attack_mode(targeted=targeted)
        self.num_iters = num_iterations
        self.loss_fn = loss_fn

    def rand_init_delta(self, delta, images, epsilon, min, max):
        delta.data.uniform_(-1, 1)
        delta.data = epsilon * delta.data
        delta.data = torch.clamp(
            images + delta.data, min=min, max=max) - images
        return delta.data

    def attack(self, images, labels, **kwargs):
        """
        :param images: The input image whose adverserial example needs to be found
        :param labels: The original label of the input. If targeted attack, should be the target label
        :return: An adversarial instance
        """

        labels = labels.to(self.device)
        images = images.to(self.device)

        if isinstance(self.loss_fn, torch.nn.BCEWithLogitsLoss):
            # Converting labels into float as required for BCELoss
            labels = labels.to(torch.float)

        # create a randomly initialized sample for PGD
        # This random initialization differentiates it from BIM
        delta = torch.zeros_like(images)
        self.rand_init_delta(delta, images, self.epsilon, self.min_val, self.max_val)
        delta.data = torch.clamp(images + delta.data, min=self.min_val, max=self.max_val) - images
        delta.requires_grad_(True)

        for i in range(self.num_iters):
            outputs = self.model(images + delta)

            loss = self._targeted * self.loss_fn(outputs, labels).to(self.device)

            # Clearing off gradients from the model
            self.model.zero_grad()

            loss.backward()

            grad_sign = delta.grad.data.sign()

            delta.data = delta.data + self.alpha * grad_sign
            delta.data = torch.clamp(delta.data, min=-self.epsilon, max=self.epsilon)
            delta.data = torch.clamp(images.data + delta.data, self.min_val, self.max_val) - images.data

        adv_images = torch.clamp(images + delta, self.min_val, self.max_val)

        return adv_images

    def __str__(self):
        return f"PGD with epsilon: {self.epsilon}, alpha: {self.alpha}, iterations: {self.num_iters} min_val: {self.min_val}, max_val: {self.max_val}, targeted: {self.is_targeted_attack}"
