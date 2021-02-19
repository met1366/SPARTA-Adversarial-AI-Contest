import os
from environment_setup import PROJECT_ROOT_DIR
from configparser import ConfigParser

from networks import Network
from networks.models.DummyNetwork import DummyNetwork
import torch
import torch.nn as nn

from networks.models.inception_resnet_v1 import InceptionResnetV1
from networks.models.AutoEncoderDefenceNetwork import AutoEncoderDefenceNetwork


def create_dummy_network(parser, section, checkpoint_dir):
    """
    A simple CNN based Fully Convolutional Network that acts as a simple baseline and we call it Dummy Network.
    It is relatively faster to train but has limited capacity owing to its simple structure
    :param checkpoint_dir: Default directory location to store pretrained model weights.
    :param parser: parser: parser object
    :param section: The section of network_config.ini file to read configurations from
    :return: DummyNetwork object
    """
    name = parser[section].get('name')
    num_classes = parser[section].getint('num_classes')
    print(f"Using Dummy Network")
    return DummyNetwork(num_classes=num_classes, name=name)


def create_facenet_model(parser, section, checkpoint_dir):
    """
    Creates an instance of InceptionResnetV1 which we call `FaceNet` in our code to explicitly specify its task
    :param checkpoint_dir: Default directory location to store pretrained model weights.
    :param parser: parser object
    :param section: The section of network_config.ini file to read configurations from
    :return: InceptionResnetV1/FaceNet object
    """
    name = parser[section].get('name')
    num_classes = parser[section].getint('num_classes')
    dropout_prob = parser[section].getfloat('dropout_prob')
    classify = parser[section].getboolean('classify')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained = parser[section].get('pretrained')
    model = InceptionResnetV1(name=name, pretrained=pretrained, classify=classify, num_classes=num_classes,
                              dropout_prob=dropout_prob, device=device, pre_train_wt_folder=checkpoint_dir)
    return model


def create_denoiautoenc_model(parser, section, checkpoint_dir):
    """
    Creates an instance of AutoEncoderDefenceNetwork
    :param parser: parser object
    :param checkpoint_dir: Default directory location to store pretrained model weights.
    :param section: The section of network_config.ini file to read configurations from
    :return: AutoEncoderDefenceNetwork object
    """
    name = parser[section].get('name')
    num_classes = parser[section].getint('num_classes')
    print(f"Using AutoEncoder Defence Network")
    return AutoEncoderDefenceNetwork(num_classes=num_classes, name=name)


def create_network(checkpoint_dir=None):
    """
    Utility function to create the model object based on input configurations. It also takes care of
    handling Multi-GPU training. If multiple GPUs are present, the method returns an instance of `nn.DataParallel`
    which seamlessly trains on multiple GPUs
    :param checkpoint_dir: Default directory location to store pretrained model weights. Default: None.
    :return: Model object
    """
    parser = read_config()
    type = parser['NETWORK'].get('key', None)
    assert type is not None, 'Please update the attack_config.ini file with the type'
    if type == 'dummy':
        model = create_dummy_network(parser=parser, section=type, checkpoint_dir=checkpoint_dir)
    elif type == 'facenet':
        model = create_facenet_model(parser=parser, section=type, checkpoint_dir=checkpoint_dir)
    elif type == 'denoiautoenc':
        model = create_denoiautoenc_model(parser=parser, section=type, checkpoint_dir=checkpoint_dir)
    else:
        raise ValueError("Invalid adversarial type chosen")
    if torch.cuda.device_count() > 1:
        print("We are using ", torch.cuda.device_count(), "GPUs for the execution!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs

        # Data Parallel creates a wrapper class so the methods present in our custom class would be hidden.
        # This is a cheaper alternative to handle the issue compared to using try-catch which is a bit slower
        # Monkey-Patching works but would make require some amount of hard-coding which is never good.
        class MyDataParallel(nn.DataParallel):
            def __init__(self, *args, **kwargs):
                super(MyDataParallel, self).__init__(*args, **kwargs)
                self.base_attr_list = Network("").__dir__()

            def __getattr__(self, name):
                if name in self.base_attr_list:
                    return getattr(self.module, name)
                else:
                    return super().__getattr__(name)
        model = MyDataParallel(model)
    return model


def read_config():
    """
    Function for reading and parsing configurations specific to the attack type
    :return: parsed configurations
    """
    config_path = os.path.join(PROJECT_ROOT_DIR, 'networks', 'network_config.ini')
    parser = ConfigParser()
    parser.read(config_path)
    return parser


def check_model_size(model):
    """
    Utility function to check the number of parameters in the model. Prints the value in Millions
    :param model: Input model to be checked for size
    :return: None
    """
    num_params = 0
    traininable_param = 0
    for param in model.parameters():
        num_params += param.numel()
        if param.requires_grad:
            traininable_param += param.numel()
    print("[Network  Total number of parameters : %.3f M" % (num_params / 1e6))
    print(
        "[Network  Total number of trainable parameters : %.3f M"
        % (traininable_param / 1e6)
    )


if __name__ == '__main__':
    model = create_network()
    check_model_size(model=model)


    class MyDataParallel(nn.DataParallel):
        def __init__(self, *args, **kwargs):
            super(MyDataParallel, self).__init__(*args, **kwargs)
            self.base_attr_list = Network("").__dir__()

        def __getattr__(self, name):
            if name in self.base_attr_list:
                return getattr(self.module, name)
            else:
                return super().__getattr__(name)


    model = MyDataParallel(model)
    # print(model.__dict__)
    x = torch.randn((10, 3, 256, 256))  # B, num_channel, H, W
    y = torch.randn((10, 3, 256, 256))  # B, num_channel, H, W
    output = model(x, y)
    loss = (1 - output).sum()
    loss.backward()
    model.zero_grad()
    print(model.name)
    print(model.additional_loss)
    print(type(model.additional_loss))