from configparser import ConfigParser
import os

from attacks.attack_models.BIM import BIM
from attacks.attack_models.CarliniWagnerL2Attack import CarliniWagnerL2Attack
from attacks.attack_models.FGSM import FGSM
from attacks.attack_models.PGD import PGD
from environment_setup import PROJECT_ROOT_DIR, LOGDIR_PREFIX


def create_fgsm(parser, section, model, folder_root, loss_fn, save_location, epsilon, min_val, max_val):
    """
    Instance of Fast Gradient Sign Method
    :param parser: parser object
    :param section: The section from config.ini from which we need to read the configurations
    :param model: The input model to generate samples in white box scenario
    :param folder_root: The root folder to which perturbed samples would be saved
    :param loss_fn: The associated loss function depending upon the task
    :param save_location: Folder name for the perturbed samples
    :param epsilon: Allowed epsilon value of perturbation
    :param min_val: Minimum value of input
    :param max_val: Maximum value of input
    :return: FGSM object
    """
    targeted = parser[section].getboolean('targeted')
    save_sub_folder = parser[section]['save_folder']
    save_folder = os.path.join(folder_root, save_location, save_sub_folder) if save_location is not None else None
    print(f"Using Fast Gradient Sign Method Attack with epsilon: {epsilon}")
    return FGSM(model=model, epsilon=epsilon, save_folder=save_folder, loss_fn=loss_fn, targeted=targeted,
                min_val=min_val, max_val=max_val)


def create_pgd(parser, section, model, folder_root, loss_fn, save_location, epsilon, min_val, max_val):
    """
    Instance of Projected Gradient Descent Method
    :param parser: parser object
    :param section: The section from config.ini from which we need to read the configurations
    :param model: The input model to generate samples in white box scenario
    :param folder_root: The root folder to which perturbed samples would be saved
    :param loss_fn: The associated loss function depending upon the task
    :param save_location: Folder name for the perturbed samples
    :param epsilon: Allowed epsilon value of perturbation
    :param min_val: Minimum value of input
    :param max_val: Maximum value of input
    :return: PGD object
    """
    alpha = parser[section].getfloat('alpha')
    targeted = parser[section].getboolean('targeted')
    num_iterations = parser[section].getint('num_iterations')
    save_sub_folder = parser[section]['save_folder']
    save_folder = os.path.join(folder_root, save_location, save_sub_folder) if save_location is not None else None
    print(
        f"Using Projected Gradient Descent Attack with epsilon: {epsilon}, alpha: {alpha}, iterations: {num_iterations}")
    return PGD(model=model, epsilon=epsilon, save_folder=save_folder,
               loss_fn=loss_fn, num_iterations=num_iterations, alpha=alpha, targeted=targeted, min_val=min_val,
               max_val=max_val)


def create_bim(parser, section, model, folder_root, loss_fn, save_location, epsilon, min_val, max_val):
    """
    Instance of Basic Iterative Method
    :param parser: parser object
    :param section: The section from config.ini from which we need to read the configurations
    :param model: The input model to generate samples in white box scenario
    :param folder_root: The root folder to which perturbed samples would be saved
    :param loss_fn: The associated loss function depending upon the task
    :param save_location: Folder name for the perturbed samples
    :param epsilon: Allowed epsilon value of perturbation
    :param min_val: Minimum value of input
    :param max_val: Maximum value of input
    :return: BIM object
    """
    alpha = parser[section].getfloat('alpha')
    targeted = parser[section].getboolean('targeted')
    num_iterations = parser[section].getint('num_iterations')
    save_sub_folder = parser[section]['save_folder']
    save_folder = os.path.join(folder_root, save_location, save_sub_folder) if save_location is not None else None
    print(f"Using Basic Iterative Method Attack with epsilon: {epsilon}, alpha: {alpha}, iterations: {num_iterations}")
    return BIM(model=model, epsilon=epsilon, save_folder=save_folder,
               loss_fn=loss_fn, num_iterations=num_iterations, alpha=alpha, targeted=targeted, min_val=min_val,
               max_val=max_val)


def create_cw(parser, section, model, folder_root, save_location, epsilon, min_val, max_val, task_type):
    """
    Instance of Carlini Wagner Method
    :param parser: parser object
    :param section: The section from config.ini from which we need to read the configurations
    :param model: The input model to generate samples in white box scenario
    :param folder_root: The root folder to which perturbed samples would be saved
    :param loss_fn: The associated loss function depending upon the task
    :param save_location: Folder name for the perturbed samples
    :param epsilon: Allowed epsilon value of perturbation
    :param min_val: Minimum value of input
    :param max_val: Maximum value of input
    :return: CarliniWagnerL2Attack object
    """
    learning_rate = parser[section].getfloat('learning_rate')
    max_iterations = parser[section].getint('max_iterations')
    binary_search_steps = parser[section].getint('binary_search_steps')
    initial_const = parser[section].getfloat('initial_const')
    targeted = parser[section].getboolean('targeted')
    abort_early = parser[section].getboolean('abort_early')
    save_sub_folder = parser[section]['save_folder']
    save_folder = os.path.join(folder_root, save_location, save_sub_folder) if save_location is not None else None

    print(f"Using CarliniWagnerL2Attack Attack with max_iterations: {max_iterations}, abort_early {abort_early}")
    return CarliniWagnerL2Attack(model=model, save_folder=save_folder, learning_rate=learning_rate,
                                 max_iterations=max_iterations, abort_early=abort_early, initial_const=initial_const,
                                 binary_search_steps=binary_search_steps, targeted=targeted, min_val=min_val,
                                 max_val=max_val, task_type=task_type)


def get_adversarial(model, args, loss_fn, epsilon, save_location=None, type=None, min_val=0, max_val=1):
    """
    Function to return instance of the correct adversarial generation technique
    :param model: The model to generate samples on in white-box setting
    :param args: argparse object for additional configurations
    :param loss_fn: Associated loss function for sample generation depending upon kind of task
    :param epsilon: Maximum perturbation allowed
    :param save_location: Parent folder location to save the adversarial samples. Default: None
    :param type: Type of attack method. If not specified, read from config.ini . Default: None
    :param min_val: Minimum value of the input samples. Default:0
    :param max_val: Maxiumum value of the input samples. Default:1
    :return: Perturbation object
    """
    parser = read_config()
    type = parser['ADVERSERIAL'].get('name', None) if type is None else type
    folder_root = os.path.join(LOGDIR_PREFIX, args.output_dir)
    assert type is not None, 'Please update the attack_config.ini file with the type'
    if type == 'fgsm':
        return create_fgsm(parser=parser, section=type, model=model, folder_root=folder_root, loss_fn=loss_fn,
                           save_location=save_location, epsilon=epsilon, min_val=min_val, max_val=max_val)
    elif type == 'pgd':
        return create_pgd(parser=parser, section=type, model=model, folder_root=folder_root, loss_fn=loss_fn,
                          save_location=save_location, epsilon=epsilon, min_val=min_val, max_val=max_val)
    elif type == 'bim':
        return create_bim(parser=parser, section=type, model=model, folder_root=folder_root, loss_fn=loss_fn,
                          save_location=save_location, epsilon=epsilon, min_val=min_val, max_val=max_val)
    elif type == 'cw':
        print("Carlini and Wagner uses its own loss function. Ignoring task specific loss fn")
        return create_cw(parser=parser, section=type, model=model, folder_root=folder_root, save_location=save_location,
                         epsilon=epsilon, min_val=min_val, max_val=max_val, task_type=args.task_type)
    else:
        raise ValueError("Invalid adversarial type chosen")


def read_config():
    """
    Function for reading and parsing configurations specific to the attack type
    :return: parsed configurations
    """
    config_path = os.path.join(PROJECT_ROOT_DIR, 'attacks', 'attack_config.ini')
    parser = ConfigParser()
    parser.read(config_path)
    return parser
