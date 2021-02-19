"""
A file which would go ahead and store specifics such as type of loss function, dataloader etc
depending upon the task. In short, any task related configuration would be handled by it.
"""
from dataset import get_alteration_dataloader, AttributesDataset
from dataset.FaceReIdDataset import get_reid_data_loader, FaceReIdDataset
import torch.nn as nn


class Task:

    def __init__(self, args, perturb_dir=None):
        self.args = args
        self.perturb_dir = perturb_dir
        self.init_config()

    def init_config(self):
        """
        Set up parameters based on the kind of task
        :return: None
        """
        if self.args.task_type == 'attr':
            # We may load the values before or after generating the samples
            self.dataloader_train, self.dataloader_val, self.dataloader_test = self.create_attribute_dataloader()
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.pred_acc = AttributesDataset.pred_acc
            self.task_type = 'attr'

        elif self.args.task_type == 'reid':
            # In case of using MTCNN some processing needs to be done on GPU before loading the data.
            # Hence we set `num_workers = 0`. If MTCNN is not used in your model, this value can be changed.
            self.dataloader_train, self.dataloader_val, self.dataloader_test = self.create_reid_dataloader()
            self.loss_fn = nn.CrossEntropyLoss()
            self.pred_acc = FaceReIdDataset.pred_acc
            self.task_type = 'reid'

        else:
            raise ValueError('Unsupported Task Type selected')

        self.min_val = self.dataloader_test.dataset.min_val
        self.max_val = self.dataloader_test.dataset.max_val

    def create_reid_dataloader(self):
        dataloader_train = get_reid_data_loader(self.args.batch_size, split='train', use_mtcnn=True,
                                                transform=None, shuffle=False, num_workers=0,
                                                dataset_min_val=-1, dataset_max_val=1)
        dataloader_val = get_reid_data_loader(self.args.batch_size, split='valid', use_mtcnn=True,
                                              transform=None, shuffle=False, num_workers=0,
                                              dataset_min_val=-1, dataset_max_val=1)
        dataloader_test = get_reid_data_loader(self.args.batch_size, split='test', use_mtcnn=True,
                                               transform=None, shuffle=False, num_workers=0,
                                               dataset_min_val=-1, dataset_max_val=1)
        return dataloader_train, dataloader_val, dataloader_test

    def create_attribute_dataloader(self):
        dataloader_train = get_alteration_dataloader(self.args.batch_size, split='train',
                                                     transform=None, shuffle=True, num_workers=4,
                                                     dataset_min_val=0, dataset_max_val=1)
        dataloader_val = get_alteration_dataloader(self.args.batch_size, split='valid',
                                                   transform=None, shuffle=False, num_workers=4,
                                                   dataset_min_val=0, dataset_max_val=1)
        dataloader_test = get_alteration_dataloader(self.args.batch_size, split='test',
                                                    transform=None, shuffle=False, num_workers=4,
                                                    dataset_min_val=0, dataset_max_val=1)
        return dataloader_train, dataloader_val, dataloader_test
