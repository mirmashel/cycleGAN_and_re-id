from io import BytesIO

# import lmdb
from PIL import Image
from torch.utils.data import Dataset
import os
import torchvision.transforms.functional as trans_fn
import numpy as np



class AlignedDatasetLoader(Dataset):
    # root_from - source dataset
    # root_to - target dataset

    def __init__(self, root_from='', root_to='', transform=None, resolution = 256):

        
        self.root_dir_from = os.path.join(root_from, 'bounding_box_train')
        self.root_dir_to = os.path.join(root_to, 'bounding_box_train')

        self.transform = transform

        self.img_names_from = sorted(os.listdir(self.root_dir_from))
        self.img_names_to = sorted(os.listdir(self.root_dir_to))

        self.resolution = resolution
                             
            
    def __len__(self):
        return len(self.img_names_from)

    def __getitem__(self, idx_from):           
        img_from = Image.open(os.path.join(self.root_dir_from, self.img_names_from[idx_from]))
        img_from = trans_fn.resize(img_from, (self.resolution, self.resolution), Image.LANCZOS)
        person_id_from = int(self.img_names_from[idx_from].split('_')[0])

        idx_to = np.random.randint(len(self.img_names_to))
        img_to = Image.open(os.path.join(self.root_dir_to, self.img_names_to[idx_to]))
        img_to = trans_fn.resize(img_to, (self.resolution, self.resolution), Image.LANCZOS)

        
        if self.transform:
            img_from = self.transform(img_from)
            img_to = self.transform(img_to)

        return {'img_from': img_from, 'img_from_id': person_id_from, 'img_to': img_to}



class DatasetForClassifier(Dataset):
    # root_from - source dataset
    # root_to - target dataset

    def __init__(self, root_from='', transform=None, resolution = 8, classes = None, val_percent = 0.1):
        
        self.root_dir_from = os.path.join(root_from, 'bounding_box_train')

        self.transform = transform

        self.img_names_from = sorted(os.listdir(self.root_dir_from))

        self.resolution = resolution

        self.ids = {}
        
        # ids = set()
        self.total_ids = 0
        for name in self.img_names_from:
            person_id_from = int(name.split('_')[0])
            if person_id_from not in self.ids:
                self.ids[person_id_from] = self.total_ids
                self.total_ids += 1
         
        if classes is not None:
            temp_classes = []
            for name in self.img_names_from:
                person_id_from = int(name.split('_')[0])
                if self.ids[person_id_from] < classes:
                    temp_classes.append(name)

            self.img_names_from = temp_classes

        self.img_names_train = []
        self.img_names_val = []
        i = 0
        while i < len(self.img_names_from):
            person_id = int(self.img_names_from[i].split('_')[0])
            all_person_images = []
            while i < len(self.img_names_from) and int(self.img_names_from[i].split('_')[0]) == person_id:
                all_person_images.append(self.img_names_from[i])
                i += 1

            c_val_images = max(int(len(all_person_images) * val_percent), 1)
            self.img_names_val += all_person_images[:c_val_images]
            self.img_names_train += all_person_images[c_val_images:]

        if classes is not None:
            self.total_ids = classes

        self.mode = 'train'


    def __len__(self):
        return len(self.img_names_train) if self.mode == 'train' else len(self.img_names_val)

    def train(self):
        self.mode = 'train'

    def eval(self):
        self.mode = 'val'

    def __getitem__(self, idx_from):
        if self.mode == 'train':
            list_from = self.img_names_train
        else:
            list_from = self.img_names_val
        
        img_from = Image.open(os.path.join(self.root_dir_from, list_from[idx_from]))
        img_from = trans_fn.resize(img_from, (self.resolution, self.resolution), Image.LANCZOS)
        person_id_from = int(list_from[idx_from].split('_')[0])
        
        if self.transform is not None:
            img_from = self.transform(img_from)

        return {'img': img_from, 'img_id': self.ids[person_id_from]}

