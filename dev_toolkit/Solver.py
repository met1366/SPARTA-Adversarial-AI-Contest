"""
The Solver class will act as the wrapper class over all the different training modes and will
be responsible for running the code block.
"""
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pprint import pprint


# The solver class in its enormity
class SolverWrapper(object):
    """
    A wrapper class for the training process
    """

    def __init__(self, network, task_wrapper, metadata_dir, checkpoint_dir,
                 logdir, args, defence=None):
        self.net = network
        self.task_wrapper = task_wrapper
        self.metadata_dir = metadata_dir
        self.checkpoint_dir = checkpoint_dir
        self.lr = args.lr
        self.factor = 0.1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logdir = logdir
        self.defence = defence

    def load_snapshot(self, model_number=None):
        """
        Load a previous version of the model if there exists one
        :param model_number: The specific model number to load. Default: None
        :return: `model_number` of the model loaded
        """
        if model_number is None:
            # First load all the models that are present in the folder
            all_model_names = os.listdir(self.checkpoint_dir)
            if len(all_model_names) == 0 or len([x for x in all_model_names if self.net.name in x]) == 0:
                # The folder is empty hence no model weights to load. Just skip
                return 0
            candidates = [x for x in all_model_names if self.net.name in x]
            model_number = max(list(map(lambda x: int(x[x.find(self.net.name) + len(self.net.name): x.find('.pth')]), candidates)))
        self.net.load(model_number=model_number, checkpoint_dir=self.checkpoint_dir)
        return model_number + 1

    def snapshot(self, model_number):
        """
        Utility function to save the model state
        :param model_number: the number to be included in model name
        :return:
        """
        # Store the model snapshot
        self.net.save(model_number=model_number, checkpoint_dir=self.checkpoint_dir)

    def construct_optimizer(self):
        """
        Creates an optimizer for the training purpose
        :return: torch.optim object
        """
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)

    def create_criterion(self):
        """
        The loss criterion for evaluation
        :return:
        """
        self.criterion = self.task_wrapper.loss_fn

    def compute_loss(self, output, label):
        loss_ = self.criterion(output, label)
        # The defence can have additional loss eg autoencoder defence that we used
        loss_ += self.defence.additional_defence_loss()
        return loss_

    @staticmethod
    def pred_acc(task_wrapper, prediction, gt_label):
        """
        A wrapper function to get hold of accuracy criterion
        :param task_wrapper: task_wrapper object to help in easy computation
        :param prediction: The predictions made by model
        :param gt_label: ground truth labels
        :return: prediction score
        """
        return task_wrapper.pred_acc(prediction=prediction, gt_label=gt_label)

    def execute_train_val(self, num_epoch, snapshot_index=None):
        """
        The method responsible for executing train and validation loops
        :param snapshot_index: Specific model instance to load. Default: None
        :param num_epoch: Number of epochs for the model to run
        :return: None
        """
        # save network structure to data folder
        with open(os.path.join(self.metadata_dir, 'nn.txt'), 'w') as file:
            file.write(str(self.net))
        # construct optimizer
        self.construct_optimizer()
        self.create_criterion()
        # Create the logger files
        self.logger_train = SummaryWriter(os.path.join(self.logdir, 'train'))
        self.logger_val = SummaryWriter(os.path.join(self.logdir, 'val'))
        if self.defence.is_perturbed:
            self.logger_adv = SummaryWriter(os.path.join(self.logdir, 'adv'))
        model_restart_offset = self.load_snapshot(snapshot_index)
        self.net.to(self.device)
        print("Execution started")
        max_val_acc = 0
        for epoch in range(num_epoch):
            # Check if the learning rate needs to be adjusted
            train_acc = self._train_model(epoch)
            val_acc, adv_acc = self._val_model(epoch)
            print("Epoch Summary:-")
            self.log_print(self.logger_train, iter=epoch, value=train_acc, mode="train", metric='accuracy')
            self.log_print(self.logger_val, iter=epoch, value=val_acc, mode="valid", metric='accuracy')
            if self.defence.is_perturbed:
                self.log_print(self.logger_adv, iter=epoch, value=adv_acc, mode="adv", metric='accuracy')
            # Currently we save the model which performs best on the validation set
            # however, this can be updated in future
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                self.snapshot(epoch + model_restart_offset)
            print("epoch {} complete".format(epoch))
        self.logger_train.close()
        self.logger_val.close()

    def _train_model(self, epoch):
        """
        The method responsible for executing one batch of training.
        Performs a lr_scheduling after every 100k iterations
        :param epoch: The current epoch of training
        :return: training accuracy
        """
        self.net.train()
        print(f"starting training epoch {epoch}")
        running_loss = 0
        running_acc = 0
        total = 0
        for iter, (_, image, label) in enumerate(tqdm(self.task_wrapper.dataloader_train)):
            # Converting labels into float since that is required by Pytorch structure
            if isinstance(self.task_wrapper.loss_fn, torch.nn.BCEWithLogitsLoss):
                # Converting labels into float as required for BCELoss
                label = label.to(torch.float)
            image, label = image.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            output = self.defence(self.net, image, label)
            loss = self.compute_loss(output, label)
            running_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            running_acc += self.pred_acc(task_wrapper=self.task_wrapper, prediction=output, gt_label=label)
            total += label.size(0)

            # Display training information
            if iter % 1000 == 999:
                self.log_print(logger=self.logger_train, iter=epoch * len(self.task_wrapper.dataloader_train) + iter,
                               value=running_loss / 1000, mode='train', metric='iter_loss')
                running_loss = 0
            # lr scheduling once we have sufficient number of iterations
            if (epoch * len(self.task_wrapper.dataloader_train) + iter) % 10000 == 9999:
                self.scale_lr()
        accuracy_value = running_acc / total
        return accuracy_value

    def _val_model(self, epoch):
        """
        The method responsible validation.
        :return: validation accuracy, Optional: Adversarial Accuracy if `Adversarial Training` used
        """
        self.net.eval()
        print(f"starting validation")
        running_loss = 0
        running_acc_val = 0
        adv_acc = 0
        total = 0
        adv_loss = 0
        # We can not use torch.no_grad() here since adversarial training would require
        # generating adversarial samples and hence, gradient flow would be needed
        for iter, (_, image, label) in enumerate(tqdm(self.task_wrapper.dataloader_val)):
            if isinstance(self.task_wrapper.loss_fn, torch.nn.BCEWithLogitsLoss):
                # Converting labels into float as required for BCELoss
                label = label.to(torch.float)
            image, label = image.to(self.device), label.to(self.device)
            output = self.net(image)
            loss = self.compute_loss(output, label)
            running_loss += loss.item()
            running_acc_val += self.pred_acc(task_wrapper=self.task_wrapper, prediction=output, gt_label=label)
            # The part where adversarial training is included
            if self.defence.is_perturbed:
                adv_output = self.defence(self.net, image, label)
                minibatch_loss = self.compute_loss(adv_output, label)
                adv_loss += minibatch_loss.item()
                adv_acc += self.pred_acc(task_wrapper=self.task_wrapper, prediction=adv_output, gt_label=label)
            total += image.size(0)
        self.log_print(self.logger_val, iter=epoch, value=running_loss / iter, mode="valid",
                       metric='loss')  # each minibatch loss is already averaged
        if self.defence.is_perturbed:
            self.log_print(self.logger_adv, iter=epoch, value=adv_loss / iter, mode="adv", metric='loss')
        accuracy_value = running_acc_val / total
        adv_accuracy = adv_acc / total
        return accuracy_value, adv_accuracy

    def scale_lr(self):
        """
        Scale the learning rate of the optimizer
        :return: in-place update of parameters
        """
        self.lr = self.lr * self.factor
        print(f"Updating the learning rate to {self.lr}")
        optimizer = self.optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr

    def log_print(self, logger, iter, value, mode, metric):
        """
        The logging function utility
        :param logger: train/val logger function used
        :param iter: current iteration count
        :return: None
        """
        logger.add_scalar(metric, value, iter)
        pprint(f'{mode}>>> {metric} : {value} >>> learning rate: {self.lr} >>> iteration: {iter}')

    @staticmethod
    def test(net, task_wrapper, device):
        """
        Method to check the model performance on the `Test` split
        :param net: The model to use
        :param task_wrapper: A wrapper object for ease in computation
        :param device: cpu/gpu
        :return: None
        """
        net.eval()
        print(f"starting final evaluation")
        acc_val = 0
        total = 0
        with torch.no_grad():
            for iter, (_, image, label) in enumerate(tqdm(task_wrapper.dataloader_test)):
                if isinstance(task_wrapper.loss_fn, torch.nn.BCEWithLogitsLoss):
                    # Converting labels into float as required for BCELoss
                    label = label.to(torch.float)
                total += image.size(0)
                image, label = image.to(device), label.to(device)
                output = net(image)
                acc_val += SolverWrapper.pred_acc(task_wrapper=task_wrapper, prediction=output, gt_label=label)

        final_acc = acc_val / total
        pprint(f'final accuracy {acc_val} / {total} : {final_acc}')
