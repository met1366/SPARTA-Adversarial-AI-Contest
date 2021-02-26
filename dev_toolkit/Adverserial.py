"""
The file has methods useful for generating adversarial examples.
"""
import time

import torch
from tqdm import tqdm

# This Target label will be a different one in our final evaluation. 
FIXED_TARGET_LABEL = 17

# During final evaluation, we would have four hours to perform the entire evaluation.
# We provide the code here on a per epoch basis which is for the particpants to play 
# around and find potential bottle-necks.
MAX_ALLOWED_EPOCH_TIME = 3000 # 50 min x 60 s = 3000 s

class AdversarialRunner:
    def __init__(self, task_wrapper, model, device, attack_method, save=False):
        super(AdversarialRunner, self).__init__()
        self.task_wrapper = task_wrapper
        self.model = model
        self.device = device
        self.attack_method = attack_method
        self.save = save

    def compute_acc(self, prediction, gt_label):
        """
        A wrapper class for getting accuracy based on the kind of dataset used
        :param prediction: logit values from the model
        :param gt_label: corresponding ground truth labels
        :return: accuracy
        """
        return self.task_wrapper.pred_acc(prediction=prediction, gt_label=gt_label)

    def get_unperturbed_acc(self):
        """
        Returns the accuracy of the model on unperturbed images
        :return: accuracy value
        """
        correct = 0
        total = 0
        self.model.eval()  # Set the model in eval mode
        with torch.no_grad():
            for data, target in self.task_wrapper.dataloader_test:
                # Send the data and label to the device
                data, target = data.to(self.device), target.to(self.device)
                # Forward pass the data through the model
                pred = self.model(data)
                # Get the count of all the matches
                correct += self.compute_acc(prediction=pred, gt_label=target)
                # Also get hold of the total count
                total += data.shape[0]
        return correct / total

    def get_perturbed_acc(self, max_eps, max_allowed_time=MAX_ALLOWED_EPOCH_TIME):
        """
        Method used to get the final accuracy value for the perturbed samples.
        :param max_eps: The maximum perturbation allowed
        :param max_allowed_time: if exceeded, all predictions are assigned same value as the unperturbed prediction
        :return:
        """
        orig_correct = 0
        perturbed_correct = 0
        total = 0
        total_penalty = 0
        perturbation_amt = 0
        self.model.eval()

        for idx, (img_name, data, target) in enumerate(tqdm(self.task_wrapper.dataloader_test)):
            total += data.size(0)
            # Send the data and label to the device
            data, target = data.to(self.device), target.to(self.device)
            orig_gt_label = target
            if self.attack_method.is_targeted_attack:
                target = FIXED_TARGET_LABEL  * torch.ones_like(target)  
            # Forward pass the data through the model
            pred = self.model(data)
            # Get the count of all the matches
            orig_correct += self.compute_acc(prediction=pred, gt_label=target)
            # Generate the perturbed sample
            start_time = time.time()
            perturbed_data = self.attack_method.generate_perturbed_image(data, target)
            # Let us record the amount of perturbation which occurred here
            perturbation_amt += torch.abs(perturbed_data - data).sum().item()
            duration = time.time() - start_time
            mask = torch.linalg.norm((data.view(data.size(0), -1) - perturbed_data.view(data.size(0), -1)),
                                     ord=float('inf'), dim=1) <= max_eps + 1e-5
            """
            The attacker has a tendency to decrease the accuracy as much as possible. To penalize the attacker, we would
            take measures that would increase the accuracy values.  
            """

            filtered_target = target
            if duration >= max_allowed_time:
                # Since time exceeded provide a value which can not match
                print("Ignoring the samples as it exceeds maximum allowed duration!")
                perturb_pred = pred
            elif mask.sum().item() == 0:
                # All samples in the minibatch exceed threshold. So, we have to ignore all the samples. We penalize
                # them simply by taking the predictions on the actual samples. This way, decrease in accuracy would be
                # minimized.
                perturb_pred = pred
            else:
                # Remove the terms which exceed this threshold
                filtered_target = target[mask]
                filtered_gt_label = orig_gt_label[mask]
                perturbed_data = perturbed_data[mask]
                # Re-classify the perturbed image
                perturb_pred = self.model(perturbed_data)
                additional_penalty = torch.logical_not(mask).sum().item()
                total_penalty += additional_penalty
                if self.save:
                    if mask.sum().item() < data.size(0):
                        # Some of the samples exceed the threshold and need to be removed.
                        # We need to check if we had masked some values and as such would need to include
                        # them during saving.
                        penalty_images = data[~mask]
                        mix_img = img_name[mask]
                        mix_gt = torch.cat((filtered_gt_label, orig_gt_label[~mask]), dim=0)
                        # Now add original elements at the end as a penalty
                        mix_perturb_data = torch.cat((perturbed_data, penalty_images), dim=0)
                        mix_img_name = torch.cat((mix_img, img_name[~mask]), dim=0)
                        mix_pred = torch.cat((pred[mask], pred[~mask]), dim=0)
                        self.attack_method.save_perturbed_images(pertrub_pred=mix_pred, orig_pred=mix_pred,
                                                                 perturbed_data=mix_perturb_data, names=mix_img_name,
                                                                 ground_truth=mix_gt, orig_data=data,
                                                                 task_type=self.task_wrapper.task_type)
                    else:
                        # Everything looks fine. We can save all the samples
                        self.attack_method.save_perturbed_images(pertrub_pred=perturb_pred, orig_pred=pred,
                                                                 perturbed_data=perturbed_data, names=img_name,
                                                                 ground_truth=orig_gt_label, orig_data=data,
                                                                 task_type=self.task_wrapper.task_type)

            perturbed_correct += self.compute_acc(prediction=perturb_pred,
                                                  gt_label=filtered_target)

        perturbed_acc = perturbed_correct / (total - total_penalty)
        original_acc = orig_correct / total
        penalty = total_penalty / total
        print("Method specs \t {}".format(self.attack_method))
        print("Original Test Accuracy = {} / {} = {:.2f}".format(orig_correct, total, original_acc))
        print("Perturbed Test Accuracy = {} / {} = {:.2f}".format(perturbed_correct, total - total_penalty, perturbed_acc))
        print("Additional penalty incurred is {:.2f}".format(penalty))
        print(f"Amount of perturbation is {perturbation_amt}")
        # Return the accuracy and an adversarial example
        return original_acc, perturbed_acc + penalty
