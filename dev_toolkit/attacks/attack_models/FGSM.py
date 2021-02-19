import torch
from attacks.attack_models.AbstractAttack import AbstractAttack


class FGSM(AbstractAttack):
    def __init__(self, model, save_folder, loss_fn, epsilon=0.05, targeted=False, min_val=0, max_val=1):
        super(FGSM, self).__init__(model=model, save_folder=save_folder, min_val=min_val, max_val=max_val)
        self.epsilon = epsilon
        self.set_attack_mode(targeted=targeted)
        self.loss_fn = loss_fn

    def attack(self, images, labels, **kwargs):
        """
        :param images: The input image whose adverserial example needs to be found
        :param labels: The original label of the input. If targeted attack, should be the target label
        :return: An adversarial instance
        """

        images = images.to(self.device)
        labels = labels.to(self.device)

        if isinstance(self.loss_fn, torch.nn.BCEWithLogitsLoss):
            # Converting labels into float as required for BCELoss
            labels = labels.to(torch.float)

        images.requires_grad = True
        outputs = self.model(images)

        loss = self._targeted * self.loss_fn(outputs, labels).to(self.device)

        # Clearing off gradients from the model
        self.model.zero_grad()

        loss.backward()

        grad_sign = images.grad.detach().sign()

        adv_images = images + self.epsilon * grad_sign

        # Adding clipping to maintain [0,1] range
        adv_images = torch.clamp(adv_images, min=self.min_val, max=self.max_val).detach()

        return adv_images

    def __str__(self):
        return f"FGSM with epsilon {self.epsilon} and min_val {self.min_val}, max_val {self.max_val}, targeted: {self.is_targeted_attack}"
