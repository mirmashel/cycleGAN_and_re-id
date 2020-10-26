from random import uniform, randint 
from math import sqrt
import os
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.utils as vutils
import torch
from time import time
from torchvision import transforms

class RandomErasing(object):
    def __init__(self, probability=0.4, er_area_min=0.02, er_area_max=0.4, asp_ratio=0.3,er_color=[150, 142, 129]):
        self.probability = probability
        self.erasing_area_min = er_area_min
        self.erasing_area_max = er_area_max
        self.aspect_ratio = asp_ratio
        self.erasing_color = er_color
        
    def __call__(self, img):
        if uniform(0, 1) <= self.probability:
            target_area = uniform(self.erasing_area_min, self.erasing_area_max) * img.shape[1] * img.shape[2]
            aspect_ratio = uniform(self.aspect_ratio, 1 / self.aspect_ratio)
            
            h = round(sqrt(target_area * aspect_ratio))
            w = round(sqrt(target_area / aspect_ratio))
            
            if h < img.shape[1] and w < img.shape[2]:
                top_left = (randint(0, img.shape[1] - h), randint(0, img.shape[2] - w))
                for i in range(3):
                    img[i, top_left[0]:top_left[0] + h, top_left[1]:top_left[1] + w] = self.erasing_color[i] / 255
        return img

def get_transforms(dataset_type):
    if dataset_type == "Train":
        transform_train = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.RandomCrop((246, 128)),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            RandomErasing()
        ])
        return transform_train
    else:
        transform_test = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return transform_test
    
class Dataset_loader(torch.utils.data.Dataset):
    
    def __init__(self, root='', dataset_type='Train', transform=None):

        self.dataset_type = dataset_type
        if dataset_type == 'Train':
            self.is_train = True
            self.root_dir = os.path.join(root, 'bounding_box_train')
        elif dataset_type == 'Test':
            self.is_train = False
            self.root_dir = os.path.join(root, 'bounding_box_test')
        elif dataset_type == 'Query':
            self.is_train = False
            self.root_dir = os.path.join(root, 'query')
        else:
            raise Exception('Invalid dataset type')

        self.transform = transform
        self.img_names = sorted(os.listdir(self.root_dir))

        self.data_to_id = {} # mapping from data person labels to training person labels
        self.id_to_data = {} # mapping from training person labels to data person labels
        self.total_ids = 0 # number of unique labels
        
        self.id_to_images = {} # mapping from training person label to indexes in images list
        
        for i, name in enumerate(self.img_names):
            person_id = int(name.split('_')[0])
            if person_id not in self.data_to_id:
                self.data_to_id[person_id] = self.total_ids
                self.id_to_data[self.total_ids] = person_id
                self.total_ids += 1
                
        for i, name in enumerate(self.img_names):
            person_id = int(name.split('_')[0])
            if person_id not in self.id_to_images:
                self.id_to_images[person_id] = [i]
            else:
                self.id_to_images[person_id].append(i) 
                             
            
    def __len__(self):
        return len(self.img_names)
    
    def get_random_class_image_idx(self, class_id, same_class):
        if same_class:
            l = self.id_to_images[self.id_to_data[class_id]]
            ind = torch.randint(0, len(l), (1, )).item()
            return l[ind]
        else:
            an_class = torch.randint(0, self.total_ids, (1, )).item()
            while an_class == class_id:
                an_class = torch.randint(0, self.total_ids, (1, )).item()
            l = self.id_to_images[self.id_to_data[an_class]]
            ind = torch.randint(0, len(l), (1, )).item()
            return l[ind]
    
    def __getitem__(self, idx):            
        img = Image.open(os.path.join(self.root_dir, self.img_names[idx]))
        person_id = self.img_names[idx].split('_')[0]
        camera_id = self.img_names[idx].split('_')[1][1]
        
        if self.transform:
            img = self.transform(img)
        
        if self.is_train:
            return {'image': img, 'person_id': self.data_to_id[int(person_id)], 'camera_id': int(camera_id)}
        else:
            return {'image': img, 'person_id': int(person_id), 'camera_id': int(camera_id)}




class Dataset_aligned_loader():
    
    def __init__(self, root='', dataset_type='Train', transform=None): 

        self.dataset_type = dataset_type
        self.datasets_loaders = []
        self.idx_ranges_to_dataset = []
        self.classes_ranges_to_dataset = []
        start_range_idx = 0
        start_range_class = 0
        for dataset_name in root:
            dataset = Dataset_loader(dataset_name, dataset_type, transform)
            self.datasets_loaders.append(dataset)
            self.idx_ranges_to_dataset.append(range(start_range_idx, start_range_idx + len(dataset)))
            self.classes_ranges_to_dataset.append(range(start_range_class, start_range_class + dataset.total_ids))
            start_range_idx += len(dataset)
            start_range_class += dataset.total_ids

        self.length = sum([len(dataset) for dataset in self.datasets_loaders])
        self.total_ids = sum([dataset.total_ids for dataset in self.datasets_loaders])
                             
            
    def __len__(self):
        return self.length

    def get_right_class_of_image(self, person_id, dataset_idx):
        return person_id + self.classes_ranges_to_dataset[dataset_idx].start

    def get_dataset_idx_and_class(self, person_id):
        dataset_idx = -1
        dataset_person_id = -1
        for i, classes_range in enumerate(self.classes_ranges_to_dataset):
            if person_id in classes_range:
                dataset_idx = i
                dataset_person_id = person_id - classes_range.start
        return dataset_idx, dataset_person_id

    def get_right_id_of_image(self, image_id, dataset_idx):
        return image_id + self.idx_ranges_to_dataset[dataset_idx].start

    def get_dataset_image_id(self, image_id):
        dataset_idx = -1
        dataset_image_id = -1
        for i, idx_range in enumerate(self.idx_ranges_to_dataset):
            if image_id in idx_range:
                dataset_idx = i
                dataset_image_id = image_id - idx_range.start
        return dataset_idx, dataset_image_id

    
    def get_random_class_image_idx(self, class_id, same_class):
        if same_class:
            dataset_idx, dataset_person_id = self.get_dataset_idx_and_class(class_id)
            image_id = self.datasets_loaders[dataset_idx].get_random_class_image_idx(dataset_person_id, same_class)
            return self.get_right_id_of_image(image_id, dataset_idx)
        else:
            an_class = torch.randint(0, self.total_ids, (1, )).item()
            while an_class == class_id:
                an_class = torch.randint(0, self.total_ids, (1, )).item()
            dataset_idx, dataset_person_id = self.get_dataset_idx_and_class(an_class)
            image_id = self.datasets_loaders[dataset_idx].get_random_class_image_idx(dataset_person_id, same_class)
            return self.get_right_id_of_image(image_id, dataset_idx)
    
    def __getitem__(self, idx):        
        dataset_idx, idx = self.get_dataset_image_id(idx)
        image = self.datasets_loaders[dataset_idx][idx]
        image['person_id'] = self.get_right_class_of_image(image['person_id'], dataset_idx)
        return image

def get_train_dataset_loader(root):
    if len(root) == 1:
        train_dataset = Dataset_loader(root[0], "Train", get_transforms("Train"))
    else :
        train_dataset = Dataset_aligned_loader(root, "Train", get_transforms("Train"))
    return train_dataset

def get_test_dataset_loader(root):
    test_dataset = Dataset_loader(root, "Test", get_transforms("Test"))
    return test_dataset

def get_query_dataset_loader(root):
    query_dataset = Dataset_loader(root, "Query", get_transforms("Query"))
    return query_dataset