import os

import torch

from attacks.adversarial_factory import get_adversarial
from defence.defence_models.AdversarialTraining import AdversarialTrain
from defence.defence_models.NoDefence import NoDefence
from defence.defence_models.autoencoder_defence import AutoEncoderAdversarialDefence
from environment_setup import PROJECT_ROOT_DIR
from configparser import ConfigParser

from networks.models.AutoEncoderDefenceNetwork import AutoEncoderDefenceNetwork


def create_adv_train(parser, section, net, args, loss_fn, min_val, max_val):
    """
    Creates Adversarial Training based defence method
    :param parser: parser object
    :param section: The section from config.ini from which we need to read the configurations
    :param net: The input model
    :param args: argparse object with added configurations
    :param loss_fn: Loss function associated with the task
    :param min_val: Minimum value of the inputs
    :param max_val: Maximum value of the inputs
    :return: AdversarialTrain Object
    """
    attack_type = parser[section].get('attack_type')
    epsilon = parser[section].getfloat('attack_epsilon')
    attack_method = get_adversarial(model=net, args=args, loss_fn=loss_fn, type=attack_type, epsilon=epsilon,
                                    min_val=min_val, max_val=max_val)
    defence = AdversarialTrain(attack_method=attack_method)
    print(f"Using Adversarial Training")
    return defence


def create_autoencoder_defence(parser, section, net, args, loss_fn, min_val, max_val):
    """
    Creates AutoEncoder based defence method
    :param parser: parser object
    :param section: The section from config.ini from which we need to read the configurations
    :param net: The input model
    :param args: argparse object with added configurations
    :param loss_fn: Loss function associated with the task
    :param min_val: Minimum value of the inputs
    :param max_val: Maximum value of the inputs
    :return: AutoEncoderAdversarialDefence Object
    """
    attack_type = parser[section].get('attack_type')
    epsilon = parser[section].getfloat('attack_epsilon')
    if torch.cuda.device_count() > 1:
        # Slightly different behavior with multi-GPU training
        assert isinstance(net.module, AutoEncoderDefenceNetwork), "For Autoencoder Defence, net should be of type " \
                                                                  "AutoEncoderDefenceNetwork "
    else:
        assert isinstance(net, AutoEncoderDefenceNetwork), "For Autoencoder Defence, net should be of type " \
                                                           "AutoEncoderDefenceNetwork "
    attack_method = get_adversarial(model=net, args=args, loss_fn=loss_fn, type=attack_type, epsilon=epsilon,
                                    min_val=min_val, max_val=max_val)
    defence = AutoEncoderAdversarialDefence(attack_method)
    print(f"Using AutoEncoderAdversarialDefence")
    return defence


def create_no_defence():
    """
    In this case, we do not want to use any sort of Defence mechanism
    :return:NoDefence Object
    """
    defence = NoDefence()
    print(f"Using No Defence")
    return defence


def create_defence(model, args, loss_fn, min_val, max_val):
    """
    Create an object with the defence type to be used
    :param model: Input model to be used in white-box setting
    :param args: argparse object with configurations
    :param loss_fn: The loss function to be used for the specific task
    :param min_val: Minimum value associated with the input
    :param max_val: Maximum value associated with the input
    :return: Defence object
    """
    parser = read_config()
    if not args.defence:
        return create_no_defence()
    type = parser['DEFENCE'].get('key', None)
    assert type is not None, 'Please update the defence_config.ini file with the type'
    if type == 'adv_train':
        return create_adv_train(parser=parser, section=type, net=model, args=args, loss_fn=loss_fn, min_val=min_val,
                                max_val=max_val)
    elif type == 'autoencoder':
        return create_autoencoder_defence(parser=parser, section=type, net=model, args=args, loss_fn=loss_fn,
                                          min_val=min_val, max_val=max_val)
    else:
        raise ValueError("Invalid adversarial type chosen")


def read_config():
    """
    Function for reading and parsing configurations specific to the defence type
    :return: parsed configurations
    """
    config_path = os.path.join(PROJECT_ROOT_DIR, 'defence', 'defence_config.ini')
    parser = ConfigParser()
    parser.read(config_path)
    return parser
