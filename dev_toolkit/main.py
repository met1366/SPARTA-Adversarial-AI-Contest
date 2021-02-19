"""
The file from where we start our execution. It is the entry point of the entire network
"""

import argparse
import os
import torch

from Adverserial import AdversarialRunner
from Solver import SolverWrapper
from attacks.adversarial_factory import get_adversarial
from defence.defence_factory import create_defence
from environment_setup import LOGDIR_PREFIX
from networks import network_factory
from TaskWrapper import Task

FIXED_EPSILON_VALUE = 8. / 255
PERTURB_IMAGE_SAVE_LOCATION = "perturb_samples"


def parse_args():
    """parse input arguments"""
    parser = argparse.ArgumentParser(description='adversarial ml')
    parser.add_argument('--epochs', help='number of epochs to train', default=10, type=int)
    parser.add_argument('--batch_size', help='Training batch size', type=int, default='32')
    parser.add_argument('--mode', help='train/test/adv/final default:train', type=str,
                        default='train')
    parser.add_argument('--task_type', help='reid/attr default:attr', type=str, default='attr')
    parser.add_argument('--num_workers', help='data loader threads', type=int, default=4)
    parser.add_argument('--output_dir', help='Directory to save logs and model. Default: checkpoints/', type=str,
                        default='checkpoints/')
    parser.add_argument('--model_number', help='Model to test final results on. Needed for test mode', type=int,
                        default='0')
    parser.add_argument('--lr', help='Learning rate of the model. Default 1e-3', type=float, default=1e-3)
    parser.add_argument('--defence', help='If Some defence mechanism used', action='store_true')
    args = parser.parse_args()
    return args


def train(args):
    """
    Method to train the model.
    :param args: argparse object
    :return:None
    """
    # Get the dataloaders
    train_task_wrapper = Task(args=args)

    checkpoint_dir = os.path.join(LOGDIR_PREFIX, args.output_dir, 'model_weights')
    log_dir = os.path.join(LOGDIR_PREFIX, args.output_dir, 'logs')
    metadata_dir = os.path.join(LOGDIR_PREFIX, args.output_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # load network
    net = network_factory.create_network(checkpoint_dir=checkpoint_dir)
    # we can also check for adversarial training here by using
    defence = create_defence(net, args, loss_fn=train_task_wrapper.loss_fn, min_val=train_task_wrapper.min_val, max_val=train_task_wrapper.max_val)

    sw = SolverWrapper(network=net,
                       task_wrapper=train_task_wrapper,
                       metadata_dir=metadata_dir,
                       checkpoint_dir=checkpoint_dir,
                       logdir=log_dir,
                       args=args,
                       defence=defence)
    # Put specific index such as '9' if want to load '<model_name>9.pth eg 'step_9.pth'
    sw.execute_train_val(args.epochs, snapshot_index=None)


def test(args):
    """
    :param args: argparse object containing the user defined arguments
    :return: None
    """
    test_task_wrapper = Task(args=args)
    checkpoint_dir = os.path.join(LOGDIR_PREFIX, args.output_dir, 'model_weights')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = network_factory.create_network(checkpoint_dir=checkpoint_dir)
    net.load(model_number=args.model_number, checkpoint_dir=checkpoint_dir)
    SolverWrapper.test(net=net, task_wrapper=test_task_wrapper, device=device)


def adv(args):
    """
    This method is used in order to generate adversarial examples
    :param args: argparse arguments that would be used
    :return: None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the model weights
    checkpoint_dir = os.path.join(LOGDIR_PREFIX, args.output_dir, 'model_weights')
    network = network_factory.create_network(checkpoint_dir=checkpoint_dir)
    network.load(model_number=args.model_number, checkpoint_dir=checkpoint_dir)
    network.to(device)
    adv_task_wrapper = Task(args=args)
    # Create the Adverserial object. Look at attack_config.ini file
    attack = get_adversarial(model=network, args=args, loss_fn=adv_task_wrapper.loss_fn,
                             save_location=PERTURB_IMAGE_SAVE_LOCATION, epsilon=FIXED_EPSILON_VALUE, min_val=adv_task_wrapper.min_val,
                             max_val=adv_task_wrapper.max_val)
    adversarial_runner = AdversarialRunner(task_wrapper=adv_task_wrapper, model=network, device=device,
                                           attack_method=attack, save=True)
    orig, adv = adversarial_runner.get_perturbed_acc(FIXED_EPSILON_VALUE)
    print("Actual accuracy of the model %.2f" % orig)
    print("Perturbed accuracy of the model %.2f" % adv)
    final_score = (orig - adv) * attack._targeted
    print(f"The final adversarial score is {final_score}")


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    # ------------------------------------
    # gpu
    # ------------------------------------
    torch.set_num_threads(args.num_workers)

    # ------------------------------------
    # train or eval
    # ------------------------------------
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'adv':
        adv(args)