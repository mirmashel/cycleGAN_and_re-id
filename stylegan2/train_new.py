import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from network.model import Embedder, Generator
from dataset import AlignedDatasetLoader
import os
import time






if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Unet adaptiveInstance')

    parser.add_argument('--name', type=str, help='Name of experiment')

    parser.add_argument('--source_path', type = str)
    parser.add_argument('--target_path', type = str)
    parser.add_argument('--sample_iters', type = int, default = 100)
    parser.add_argument('--save_iters', type = int, default = 10_000)

    parser.add_argument('--total_iters', type=int, default=100_000, help='number of samples used for each training phases')
    parser.add_argument('--phase', type=int, default=10_000, help='number of samples used for each training phases')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--sched', action='store_true', help='use lr scheduling')
    parser.add_argument('--pic_size', default=256, type=int, help='max image size')
    parser.add_argument('--pad_size_to', default = 256, type = int)
    parser.add_argument('--code_size', default=512, type=int)
    parser.add_argument('--n_mlp', default=None, type=int)
    parser.add_argument('--lambda_idt', default = 0, type = float)
    parser.add_argument('--lambda_prcp', default = 0, type = float)

    parser.add_argument('--path_vgg_weights', type = str, default = "vgg_weights")
    parser.add_argument('--prefix_vgg_weights', type = str, default = "VGG_13")
    parser.add_argument('--no_use_gan', action = 'store_true')

    parser.add_argument('--ckpt_name', default=None, type=str, help='load from previous checkpoints')

    args = parser.parse_args()
    args.experiment_dir = os.path.join("./checkpoints", args.name)
    os.makedirs(args.experiment_dir, exist_ok = True)

    transform = transforms.Compose(
        [
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = AlignedDatasetLoader(args.source_path, args.target_path, transform, resolution = args.pic_size)


    E = nn.DataParallel(Encoder(args.pic_size, args.pad_size_to).cuda())
    G = nn.DataParallel(Generator(args.pic_size, args.pad_size_to).cuda())
    D = nn.D