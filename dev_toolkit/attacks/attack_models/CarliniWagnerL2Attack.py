"""
Motivated from
https://github.com/BorealisAI/advertorch/blob/7f65085aca30319d245cbaec00aad29ec2820326/advertorch/attacks/carlini_wagner.py
"""
from pprint import pprint

from attacks.attack_models.AbstractAttack import AbstractAttack
import torch
import torch.nn as nn
import torch.optim as optim

REPEAT_STEP = 10
CARLINI_L2DIST_UPPER = 1e10
CARLINI_COEFF_UPPER = 1e10
INVALID_LABEL = -1
ONE_MINUS_EPS = 0.999999
PREV_LOSS_INIT = 1e6
NUM_CHECKS = 10
TARGET_MULT = 10000.0
UPPER_CHECK = 1e9


def is_successful(y1, y2, targeted, is_multi_label=False):
    if is_multi_label:
        return torch.any(y1 != y2, dim=1)
    if targeted:
        return y1 == y2
    else:
        return y1 != y2


class CarliniWagnerL2Attack(AbstractAttack):
    def __init__(self, model, save_folder, learning_rate=0.01, max_iterations=100, abort_early=True,
                 initial_const=1e-3, binary_search_steps=55555, targeted=False, min_val=0, max_val=1, task_type='attr'):
        super(CarliniWagnerL2Attack, self).__init__(model=model, save_folder=save_folder, min_val=min_val, max_val=max_val)
        self.model = model
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.binary_search_steps = binary_search_steps
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.set_attack_mode(targeted=targeted)
        self.num_classes = self.model.num_classes
        # The last iteration (if we run many steps) repeat the search once.
        self.repeat = binary_search_steps >= REPEAT_STEP
        # Call super class to set mode -> targeted vs untargeted
        self.targeted = targeted
        self.clip_min = min_val
        self.clip_max = max_val
        self.is_multi_label = task_type == 'attr'

    def attack(self, images, labels, **kwargs):
        """
        :param images: The input image whose adverserial example needs to be found
        :param labels: The original label of the input. If targeted attack, should be the target label
        :return: An adverserial instance
        """

#        labels = labels.to(self.device)
        images = images.to(self.device)
        labels = self._get_predicted_label(images=images)

        images = images.clone().detach()
        batch_size = images.size(0)
        coeff_lower_bound = images.new_zeros(batch_size)  # Initially zeros torch.zeros_like(labels).float()  #
        coeff_upper_bound = images.new_ones(batch_size) * CARLINI_COEFF_UPPER  # Very large value initially torch.ones_like(labels).float() * CARLINI_COEFF_UPPER  #
        loss_coeffs = torch.ones(batch_size).to(self.device) * self.initial_const #torch.ones_like(labels).float() * self.initial_const

        final_l2distsqs = [CARLINI_L2DIST_UPPER] * batch_size
        final_labels = torch.full(labels.shape, INVALID_LABEL)  #[INVALID_LABEL] * batch_size
        final_advs = images
        # we work on the image space and use tanh for ensuring values of examples remain within
        # the [0, 1]^n range.
        x_atanh = self._get_arctanh_x(images)
        y_onehot = self.to_one_hot(labels, self.num_classes).float()

        final_l2distsqs = torch.FloatTensor(final_l2distsqs).to(images.device)
        final_labels = torch.LongTensor(final_labels).to(images.device)

        # Start binary search
        for outer_step in range(self.binary_search_steps):
            delta = nn.Parameter(torch.zeros_like(images))
            optimizer = optim.Adam([delta], lr=self.learning_rate)
            cur_l2distsqs = [CARLINI_L2DIST_UPPER] * batch_size
            # curl_labels and cur_l2_dist is initialized with `inf`
            cur_labels = torch.full((labels.shape), INVALID_LABEL) #[INVALID_LABEL] * batch_size
            cur_l2distsqs = torch.FloatTensor(cur_l2distsqs).to(images.device)
            cur_labels = torch.LongTensor(cur_labels).to(images.device)
            prevloss = PREV_LOSS_INIT

            if self.repeat and outer_step == (self.binary_search_steps - 1):
                loss_coeffs = coeff_upper_bound
            for ii in range(self.max_iterations):
                loss, l2distsq, output, adv_img = self._forward_and_update_delta(optimizer, x_atanh, delta, y_onehot,
                                                                                 loss_coeffs)

                if self.abort_early:
                    if ii % (self.max_iterations // NUM_CHECKS or 1) == 0:
                        if loss > prevloss * ONE_MINUS_EPS:
                            break
                        prevloss = loss

                self._update_if_smaller_dist_succeed(adv_img, labels, output, l2distsq, cur_l2distsqs, cur_labels,
                                                     final_l2distsqs, final_labels, final_advs)
            self._update_loss_coeffs(labels, cur_labels, batch_size, loss_coeffs, coeff_upper_bound, coeff_lower_bound)

        return final_advs

    def __str__(self):
        return f"Carlini & Wagner with learning rate: {self.learning_rate} and num_classes: {self.num_classes} min_val: {self.min_val}, max_val: {self.max_val}, targeted: {self.is_targeted_attack}"

    def _get_arctanh_x(self, images):
        """

        :param images: Images in tanh scale
        :return: Images in input space scale
        """
        result = torch.clamp((images - self.clip_min) / (self.clip_max - self.clip_min),
                             min=0., max=1.) * 2 - 1
        return self.torch_arctanh(result * ONE_MINUS_EPS)

    def torch_arctanh(self, x):
        """
        Utility function to compute arctanh(x)
        :param x: input tensor
        :return: tanh value
        """
        return (torch.log((1 + x) / (1 - x))) * 0.5

    def to_one_hot(self, labels, num_classes=10):
        """
            Take a batch of label y with n dims and convert it to
            1-hot representation with n+1 dims.
            Link: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/24
            """
        if self.is_multi_label:
            return labels

        labels = labels.clone().detach().view(-1, 1)
        labels_one_hot = labels.new_zeros((labels.size()[0], num_classes)).scatter_(1, labels, 1)
        return labels_one_hot

    def _forward_and_update_delta(self, optimizer, x_atanh, delta, y_onehot, loss_coeffs):
        """
        Performs the optimization step of C&W
        :param optimizer: Adam optimizer used for the step
        :param x_atanh: image
        :param delta: perturbations to be apploed to image
        :param y_onehot: one hot representation of labels
        :param loss_coeffs: loss_coeff to be applied to each instance
        :return: loss, l2_square_distance, prediction and adversarial image
        """
        optimizer.zero_grad()
        # ensure that adv example withing (0,1) in values
        adv = self.tanh_rescale(delta + x_atanh, self.clip_min, self.clip_max)
        # ensure that original image withing (0,1) in values
        transimgs_rescale = self.tanh_rescale(x_atanh, self.clip_min, self.clip_max)
        output = self.model(adv)
        l2distsq = self.calc_l2distsq(adv, transimgs_rescale)
        loss = self._loss_fn(output, y_onehot, l2distsq, loss_coeffs)
        loss.backward()
        optimizer.step()

        return loss.item(), l2distsq.data, output.data, adv.data

    def tanh_rescale(self, x, x_min=-1., x_max=1.):
        """
        Ensure within min-max range of the input
        :param x: input image
        :param x_min: minimum value of images
        :param x_max: maximum value of images
        :return: rescaled values in tanh space
        """
        # tanh rescaling basically ensures that we are within the limits of min and max for delta.
        # another thing might be to clip values directly.
        return (torch.tanh(x)) * 0.5 * (x_max - x_min) + (x_max + x_min) * 0.5

    def calc_l2distsq(self, x, y):
        """
        L2 distance in the input images
        :param x: first tensor
        :param y: second tensor
        :return: element wise l2 square distance between tensors
        """
        d = (x - y) ** 2
        return d.view(d.shape[0], -1).sum(dim=1) 

    def _loss_fn(self, output, y_onehot, l2distsq, const):
        """
        Compute the loss value for the inputs
        :param output: model prediction
        :param y_onehot: one-hot representation of input labels
        :param l2distsq: l2 square distance between the tensors
        :param const: constance for scaling the max categorical loss
        :return: Final loss evaluation
        """
        if self.is_multi_label:
            ones = (y_onehot * output)
            zeros = (1.0 - y_onehot) * output
            loss1 = torch.clamp(ones - zeros, min=0.)
            loss2 = l2distsq.sum()
            loss1 = torch.sum(const * loss1.sum(dim=1))
            loss = loss1 + loss2
            return loss
        # The default case of cross entropy loss
        real = (y_onehot * output).sum(dim=1)

        other = ((1.0 - y_onehot) * output - (y_onehot * TARGET_MULT)).max(1)[0]
        # - (y_onehot * TARGET_MULT) is for the true label not to be selected

        if self.targeted:
            loss1 = torch.clamp(other - real, min=0.)
        else:
            loss1 = torch.clamp(real - other, min=0.)
        loss2 = l2distsq.sum()
        loss1 = torch.sum(const * loss1)
        loss = loss1 + loss2
        return loss

    def _update_if_smaller_dist_succeed(self, adv_img, labs, output, l2distsq, cur_l2distsqs, cur_labels,
                                        final_l2distsqs, final_labels, final_advs):

        target_label = labs
        output_logits = output
        output_label = torch.max(output_logits, 1)[1] if not self.is_multi_label else (output_logits > 0).to(torch.long).to(output_logits.device)
        # Update the final values based on possibility of perturbed samples at lesser distance
        mask = (l2distsq < cur_l2distsqs) & self._is_successful(output_logits, target_label, True)
        cur_l2distsqs[mask] = l2distsq[mask]  # redundant

        cur_labels[mask] = output_label[mask]

        # Update the final values based on possibility of perturbed samples at lesser distance
        mask = (l2distsq < final_l2distsqs) & self._is_successful(output_logits, target_label, True)
        final_l2distsqs[mask] = l2distsq[mask]
        final_labels[mask] = output_label[mask]
        final_advs[mask] = adv_img[mask]

    def _update_loss_coeffs(self, labs, cur_labels, batch_size, loss_coeffs,
                            coeff_upper_bound, coeff_lower_bound):

        for ii in range(batch_size):
            if self._is_successful(cur_labels[ii], labs[ii], False):
                coeff_upper_bound[ii] = min(coeff_upper_bound[ii], loss_coeffs[ii])

                if coeff_upper_bound[ii] < UPPER_CHECK:
                    loss_coeffs[ii] = (coeff_lower_bound[ii] + coeff_upper_bound[ii]) / 2
            else:
                coeff_lower_bound[ii] = max(coeff_lower_bound[ii], loss_coeffs[ii])
                if coeff_upper_bound[ii] < UPPER_CHECK:
                    loss_coeffs[ii] = (coeff_lower_bound[ii] + coeff_upper_bound[ii]) / 2
                else:
                    loss_coeffs[ii] *= 10

    def _is_successful(self, output, label, is_logits):
        if self.is_multi_label:
            if is_logits:
                pred = (output > 0).to(torch.long) 
            else:
                pred = output
                if torch.any(pred == INVALID_LABEL):
                    return False
                pred = pred.unsqueeze(0)
                label = label.unsqueeze(0)
        elif is_logits:
            output = output.detach().clone()
            pred = torch.argmax(output, dim=1)
        else:
            pred = output
            if pred == INVALID_LABEL:
                return pred.new_zeros(pred.shape).byte()
        return is_successful(pred, label, self.targeted, self.is_multi_label)

    def _get_predicted_label(self, images):
        with torch.no_grad():
            if self.is_multi_label:
                return (self.model(images) >= 0).to(torch.float)
            else:
                return torch.max(self.model(images), dim=1)[1]
