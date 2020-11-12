from io import BytesIO

# import lmdb
from PIL import Image
from torch.utils.data import Dataset
import os
import torchvision.transforms.functional as trans_fn
import numpy as np

class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=8):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img




class AlignedDatasetLoader(Dataset):
    # root_from - source dataset
    # root_to - target dataset

    def __init__(self, root_from='', root_to='', transform=None, resolution = 8, enc_resolution = 256):

        
        self.root_dir_from = os.path.join(root_from, 'bounding_box_train')
        self.root_dir_to = os.path.join(root_to, 'bounding_box_train')

        self.transform = transform

        self.img_names_from = sorted(os.listdir(self.root_dir_from))
        self.img_names_to = sorted(os.listdir(self.root_dir_to))

        self.resolution = resolution
        self.enc_resolution = enc_resolution

        ids = set()
        self.total_ids = 0
        for name in self.img_names_from:
            person_id_from = int(name.split('_')[0])
            if person_id_from not in ids:
                ids.add(person_id_from)
                self.total_ids += 1
                             
            
    def __len__(self):
        return len(self.img_names_from)
    
    def set_resolution(resolution):
        self.resolution = resolution

    def __getitem__(self, idx_from):           
        img_from = Image.open(os.path.join(self.root_dir_from, self.img_names_from[idx_from]))
        img_encode = trans_fn.resize(img_from, (self.enc_resolution, self.enc_resolution), Image.LANCZOS)
        img_from = trans_fn.resize(img_from, (self.resolution, self.resolution), Image.LANCZOS)
        person_id_from = int(self.img_names_from[idx_from].split('_')[0])



        idx_to = np.random.randint(len(self.img_names_to))
        img_to = Image.open(os.path.join(self.root_dir_to, self.img_names_to[idx_to]))
        img_to = trans_fn.resize(img_to, (self.resolution, self.resolution), Image.LANCZOS)

        
        if self.transform:
            img_from = self.transform(img_from)
            img_to = self.transform(img_to)
            img_encode = self.transform(img_encode)

        return {'img_from': img_from, 'img_from_id': person_id_from, 'img_enc': img_encode, 'img_to': img_to}



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

