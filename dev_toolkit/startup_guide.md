# Development toolkit for participants of adversarial SAFAIR AI contest

This is a development toolkit for the Sparta SAFAIR program.

This toolkit includes:

* Dev dataset which participants can use for development and testing of their
  attacks and defences.
* Sample adversarial attacks.
* Sample adversarial defences.
* Tool to run attacks against defences and compute score.

## Installation

### Prerequisites

Following software required to use this package:

* Python 3.6 with installed [Numpy](http://www.numpy.org/)
  and [Pillow](https://python-pillow.org/) packages.
* [Docker](https://www.docker.com/)

Additionally, all provided examples are written with use of
the [PyTorch](https://pytorch.org/).

Additionally, other utility packages required can be obtained by taking a look at the `requirements.txt` file.

### Installation procedure

The requirements are placed in the requirements.txt file. Please use

```pip install -r requirements.txt```

to set up the dependencies. We also suggest making virtual environments
by using [conda](https://docs.conda.io) or python virtual environment using
```
python3 -m venv /path/to/new/virtual/environment
```

Please update pip to the latest version else there might be some errors during installation. We take care of this while building the docker images but manual update is needed if you
are creating virtual environments.


## Dataset

This toolkit includes DEV dataset which uses the publicly available [celebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset.
DEV dataset could be used for development and testing of adversarial attacks
and defences. Final evaluation will be made on the hidden dataset which will not be accessible to the participants. 

The dataloaders provided with the dev toolkit will take care of downloading the datset. However, since the celebA dataset is
hosted on google drive automatically, often time the direct download would fail with a warning message indicating that the download
limit is exceeded. This is an expected behaviour. You can try again later to download the dataset. 

However, often times, even if the direct download fails, it is still possible to download the dataset using a browser. 
You can download the zip file and extract it in the `data` folder. Please remember that the expected structure of the folder
is

```
--- data
    --- reid_dataset
        --- train.csv
        --- val.csv
        --- test.csv
    --- celeba
        --- img_align_celeba
        --- identity_celebA.txt
        --- list_attr_celeba.txt
        --- list_bbox_celeba.txt
        --- list_eval_partition.txt
        --- list_landmarks_align_celeba.txt

```  
You can delete the zip file in order to save space. Please make sure you follow the structure closely.

The **Re-identification** task uses the images as well. Hence, before starting either of the training processes, please make
sure the images are properly downloaded. 

## Sample attacks and defences

Toolkit includes examples of attacks and defences in the following directories:

* `attacks/` - directory with examples of attacks:
  * `attacks/attack_models/fgsm/` - Fast gradient sign attack.
  * `attacks/attack_models/bim/` - Basic Iterative Method Attack
  * `attacks/attack_models/pgd/` - Projected Gradient Descent Attack
  * `attacks/attack_models/CarliniWagnerL2Attack/` - The Carlini Wagner Attack based on L2 metric
  
* `defence/` - directory with examples of defences:
  * `defence/defence_models/NoDefence` - baseline classifier,
    which actually does not provide any defence against adversarial examples.
  * `defence/defence_models/AdversarialTraining` - A class which acts as a wrapper and allows for the Adversarial training of the model
  * `defence/defence_models/autoencoder_defence` - Uses a [denoising autoencoder](https://arxiv.org/pdf/1812.03087.pdf) for the robustness.

### Structure of attacks and defences

Each attack and defence should be stored in a separate subdirectory,
should be self-contained and intended to be run inside Docker container.

One can create a new attack model by extending the `attacks/attack_models/AbstractAttack.py`. Along with this, for an easy interplay between different methods, we have
`attacks/attack_config.ini` file where one can put the required configurations. We also provide with the `attacks/adversarial_factory` file which can be used to expose the attack method. This comes in handy
during final evaluation of the methods. Please take a look at these files. A sample example from the config file is given here:

Example of `attack_config.ini`:

```
alpha=0.01
num_iterations=400
save_folder = pgd
targeted=False
```

* `save_folder` indicates the location to store generated adversarial perturbations.
* `targeted` indicates whether the mechanism uses targeted or untargeted attack.
Other attribute specific to the model can be included as well.
* `alpha` value is specific to the pgd attack
* `num_iterations` indicates the number of iterations we have to run through for the attack


The perturbations are computed using [infinity norm](https://en.wikipedia.org/wiki/Uniform_norm)

### Project Structure

In general, we structure the project as follows:-

1. The attack related configurations and model source code are present in `attacks` folder. 
2. The defence related configuratins and model source code is present in `defence` folder.
3. The pytorch dataset files are present in `dataset` folder
4. Raw data is present in `data` folder
5. All the generated tensorboard logs and saved models are present in `execution_results` folder. Even the adversarial examples that are going to be generated would be present in the `execution_results` folder
6. For loading a pretrained model, the code searches it in the `model_weights` subdirectory in the `execution_results/<output_dir>` folder (where output_dir is the specific directory name given in each execution).
 So in case you want to load a pretrained model, please make sure model weights are placed in this subfolder.

### Structure of execution results folder

Let us say you run the code with `--output_dir=adv5`, in that case you would see:-

```
--- execution_results
    --- adv5 
           --- logs
                --- adv
                --- train
                --- val
           --- model_weights
           --- nn.txt
           --- perturb_samples
                    --- fgsm
                        --- images
                        --- json
                        --- orig_images
                    --  pgd
                        --- images
                        --- json
                        --- orig_images
```

### Attack
                
To create adversarial examples, take a look at `AbstractAttack.py` in
`attacks` folder. All the Adversarial classes should be a subclass of this class.

The `attack_config.ini` file is created to enable easy configuration management. Here is a snippet of the file.
```
[ADVERSERIAL]
name = bim 

[fgsm]
save_folder = fgsm
targeted=False

[bim]
alpha = 0.01
num_iterations = 100
save_folder = bim
targeted=False
```

The `name` configuration in `[ADVERSARIAL]` section determines the active attack. 

To create a new Attack class
1. Extend from `AbstractAttack` in `attacks/attack_models/AbstractAttack.py` 
1. Create a section in the `attack_config.ini` file
2. Change the `name` in `attack_config.ini` to the new attack name 
3. Update the `adversarial_factory.py` class with instantiation of the new type
4. The generated samples can be stored by using `save=True` in `Adversarial.py` while calling the method `get_perturbed_acc()`

The generated samples can also be visualized using a simple matplotlib utility. 
`visualizations/visualize_attributes.py` file has the logic built in. This file is dependant upon the logic used for saving generated
samples in `save()` method of `attacks/AbstractAttack.py`. So any change in the method should lead to corresponding changes in the file. 

### Defence

All defence methods extend the AbstractDefence class. 
Currently we are having Adversarial Training and Denoising Autoencoder methods. For creation of the defence type we use `defence.config` file

```
[DEFENCE]
key = adv_train 

[no_defence]

[autoencoder]
attack_type=fgsm
attack_epsilon = 0.05

[adv_train]
attack_type=fgsm
attack_epsilon = 0.05
```  

This is very similar in spirit to the attack counter part.

The method used for defence would depend upon the `key` in `defence_config.ini` file. The `defence_factory.py` is executed in order to get the
correct defence mechanism.

To create a new Defence class

1. Extend from `AbstractDefence` in `defence/defence_models/base.py`
2. Create a section in the `defence_config.ini` file
3. Change the `key` in `defence_config.ini` to the new defence name 
4. Update the `defence_factory.py` class with instantiation of the new type
 
### Execution Steps

To start with execute 
>>> python main.py -h

to get a list of options and instructions to execute the program.
For starting a simple training loop, simply use
>>> python main.py --mode train --task_type attr

Since the options have default arguments in most of the cases (please check `main.py` file for the entire list), one can make use of the default options and
reduce the above command command to

>>> python main.py

since default operation mode is `train` and default task is `reid`

We support two tasks currently.
1. Targeted Re-identification
2. Facial Attribute manipulations

This can be selected by using `task_type` argument. For instance,
>>> python main.py --output_dir adv5 --task_type reid --lr 0.01

Here we specify the `output_dir` and `lr` (learning rate). We take a detailed look at the folder structure and `output_dir` role in [Structure of logs folder](#structure-of-logs-folder).


The results shall be computed on the CelebA dataset.

Once the training is done, adversarial samples can be generated using
>>> python main.py --mode adv --output_dir adv5 --model_number 9

The mode argument should be `adv` and the execution also expects a saved model to load the weights. For this it will use `output_dir` argument as passed from the command line. <br/>
For instance, if we use `--output_dir adv5` as an argument, the framework would search for the model weights within `execution_results/adv5/model_weights/`.

The models are saved with names such as `step_0.pth`. The prefix can be changed from `network_config.ini` file. This is a required parameter for the model.
The default value for `output_dir` is `checkpoints/` but since the folder is pivotal to the code execution, we strongly encourage not using the default folder name.

To execute a defence method, we would use

>>> python main.py  --output_dir adv6 --defence --task_type reid

The `--defence` switch is used to start training the model with the configured defence technique.


### Data Augmentations

Data Augmentations have proven to be really useful for improving performance of the Deep Neural Networks. We strongly encourag
the participants to explore different data augmentation techniques. PyTorch provides some built in augmentation methods which might
prove to be really useful for the training process, please take a look [here](https://pytorch.org/docs/stable/torchvision/transforms.html).

However, in general, different augmentation methods would lead to different range of values for the inputs.
Deep Learning models and attack methods are **very sensitive** to the range of these inputs values. Hence, one needs to be really careful with
the augmentation techniques. To make things easier, we provide the `TaskWrapper.py` class which can be used to handle it.

```
my_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
dataloader_test = get_reid_data_loader(self.args.batch_size, split='test', use_mtcnn=True,
                                               transform=my_transform, shuffle=False, num_workers=0,
                                               dataset_min_val=0, dataset_max_val=1)

``` 

We explicitly provide the **minimum and maximum** value that the dataset is expected to have. For instance, the `ToTensor()`
methods scales the inputs in the range of `[0, 1]` and the values are explicitly passed to the dataloader. this encourages
seamless interaction between the attack mechanisms. Please make sure that you update the value range in case you want to use
different augmentation techniques.

### Multiple GPU support

The code can be run on multiple GPUs without any configuration change needed. The 
code detects for presence of multiple GPUs and if present, it can handle the multi-processing
itself. Similarly, the code handles execution on CPU seamlessly.

### Summary
We include the following attacks in the `attacks/attack_models` folder.
1. FGSM -> Fast Gradient Sign Method
2. PGD -> Projected Gradient Descent
3. CW -> Carlini and Wagner L2 Attack
4. BIM -> Basic Iterative Method

We have the following defence methods in `defence/defence_models`.
1. AdversarialTraining
2. autoencoder_defence -> De-noising Autoencoder based defence.

Visualization utility based on matplotlib is present in `visualizations` folder.

A sample Dockerfile is provided which shows aids in creating the docker images. For building the image, one can use:-

>>> docker build -t sparta_image:1.0 --build-arg USER_ID=<some_user_id> --build-arg GROUP_ID=<user's group id> .

Here `some_user_id` and `user's group id` is to ensure that the docker container does not run with ROOT privileges. Please 
ensure that the user has GPU privileges. Once the container is built, it can be run using

>>> docker run --gpus all --ipc=host --rm -it -v ${PWD}/data/:/app/data/ -v ${PWD}/execution_results:/app/execution_results sparta_image:1.0 --mode train --task_type reid --output_dir adv5


### Docker 
The code snippets submitted would be run as a Docker container. This ensures easy dependency 
management. The participants would submit their `Dockerfile` along with the code. The docker container
would be built by us. We would be using `nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04` as our base image.
Participants are encouraged to use the same base image since it is compatible with our infrastructure.

Please install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) if tou want to run CUDA commands
for evaluation. The default container would expose `main.py` file hence running of the code remains pretty similar
to what we have outlined above.

A sample Dockerfile is provided which shows aids in creating the docker images. For building the image, one can use:-

>>> docker build -t sparta_image:1.0 --build-arg USER_ID=<some_user_id> --build-arg GROUP_ID=<user's group id> .

Here `some_user_id` and `user's group id` is to ensure that the docker container does not run with ROOT privileges. Please 
ensure that the user has GPU privileges. Once the container is built, it can be run using

>>> docker run --gpus all --ipc=host --rm -it -v ${PWD}/data/:/app/data/ -v ${PWD}/execution_results:/app/execution_results sparta_image:1.0 --mode train --task_type reid --output_dir adv5

 * Here, we allow GPU access to the containers by using `--gpus all` and allow for memory sharing with the host machine
by using `--ipc=host`.These two steps are essential.
 * We also mount the `data` folder to allow access to dataset
and finally, mount `execution_results` folder so that we can have access to all generated logs, model_weights and perturbed_samples.
* `sparta_image:1.0` indicates the docker image and version number.
* `--mode train --task_type reid --output_dir adv5` Indicate the task and mode for the code execution.