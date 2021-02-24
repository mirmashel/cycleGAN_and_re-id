from random import uniform, randint 
from math import sqrt
import os
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.utils as vutils
import torch
from time import time
from torchvision import transforms


class DukeStyleDataset():
    
    def __init__(self, root='', dataset_type='Train', label_shift = 0):

        if dataset_type == 'Train':
            self.root_dir = os.path.join(root, 'bounding_box_train')
            self.dataset = self.load_dir(self.root_dir, True, label_shift)

        elif dataset_type == 'Test':
            self.root_dir = os.path.join(root, 'bounding_box_test')
            self.dataset = self.load_dir(self.root_dir, False)

        elif dataset_type == 'Query'
            self.root_dir = os.path.join(root, 'query')
            self.dataset = self.load_dir(self.root_dir, False)

        else:
            raise Exception('Invalid dataset type')

    def load_dir(self, root, re_label, label_shift = 0):
        img_names = sorted(os.listdir(root))

        self.pid_to_label = {} # mapping from data person labels to training person labels
        self.total_ids = label_shift # number of unique labels

        dataset = []
        for img_name in img_names:
            person_id = self.img_names[idx].split('_')[0]
            camera_id = self.img_names[idx].split('_')[1][1]
            img_path = os.path.join(root, img_name)
            if person_id not in self.pid_to_label:
                self.pid_to_label[person_id] = self.total_ids
                self.total_ids += 1
            if re_label:
                person_id = self.pid_to_label(person_id)
            dataset.append((img_path, person_id, camera_id))

        return dataset


class MultiDukeStyleDataset():

    def __init__(self, roots = []):
        self.dataset = []
        self.total_ids = 0
        for root in roots:
            new_dataset = DukeStyleDataset(root, 'Train', self.total_ids)
            self.dataset += new_dataset.dataset
            self.total_ids += new_dataset.total_ids


def get_train_dataset_loader(root):
    if len(root) == 1:
        train_dataset = DukeStyleDataset(root[0], "Train")
    else :
        train_dataset = MultiDukeStyleDataset(root, "Train")
    return train_dataset

def get_val_dataset_loader(root):
    test_dataset = DukeStyleDataset(root, "Val")
    return test_dataset
