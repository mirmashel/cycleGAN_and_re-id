import argparse
import os
import os.path as osp
import torch

class Options():

    def __init__(self, is_train = True):
        if is_train:
            parser = argparse.ArgumentParser(description = "Train options")
            parser.add_argument('--nepochs', type = int, default = 256, help = '# of epochs')
            parser.add_argument('--start_epoch', type = int, default = 1, help = '# of start epoch')
            parser.add_argument('--batch_size', type = int, default = 16, help = 'Batch size')
            parser.add_argument('--lr', type = float, default = 0.05, help = 'Initial learning rate for SGD optimizer')
            parser.add_argument('--warmup_epoch', type = int, default = 0, help = 'Epochs to GradualWarmupScheduler')
            parser.add_argument('--start_step_lr', type = int, default = 40, help = 'Epoch from which reduce lr')

            parser.add_argument('--dataroot', type = str, action = 'append', required = True, help = 'list of paths to images root(Duke style)')
            parser.add_argument('--gpu_id', type = int, default = 0, help = 'Specify gpu ids if -1 than CPU')
            parser.add_argument('--name', type = str, required = True, help = 'Name of the experiment')
            parser.add_argument('--save_suffix', type = str, default = '', help = 'Suffix to save results')

            parser.add_argument('--only_backbone', action = 'store_true', help = 'Need if load only backbone weights (for finetuning on another number of person_id)')
            parser.add_argument('--initial_weights', type = str, help = 'Initialization weights experiment name')
            parser.add_argument('--initial_suffix', type = str, default = 'latest', help = 'Intital suffix')
            parser.add_argument('--pretrain_classifiers_epochs', type=int, default = 0, help = 'Steps to pretrain classifiers')
            parser.add_argument('--pretrain_classifiers_lr', type=int, default = 0.05, help = 'Steps to pretrain classifiers')

            parser.add_argument('--checkpoint_every', type = int, default = 1000, help = 'Number of epochs to make checkpoint')
            parser.add_argument('--log_file', type = str, default = '', help = 'file to save logs')



            self.opt = parser.parse_args()

            self.opt.save_weights_path = osp.join("./checkpoints", self.opt.name)

            if self.opt.log_file != '':
                self.opt.log_file = osp.join(self.opt.save_weights_path, self.opt.log_file)

            if not osp.exists(self.opt.save_weights_path):
                os.makedirs(self.opt.save_weights_path)
            self.opt.load_weights_path = osp.join("./checkpoints", self.opt.initial_weights) if self.opt.initial_weights is not None else None

            self.opt.device = torch.device(self.opt.gpu_id) if (torch.cuda.is_available() and self.opt.gpu_id >= 0) else torch.device("cpu")

        else:
            parser = argparse.ArgumentParser(description = "Test options")

            parser.add_argument('--dataroot', type = str, action = 'store', required = True, help = 'path to images root(Duke style)')
            parser.add_argument('--gpu_id', type = int, default = 0, help = 'Specify gpu ids if -1 than CPU')

            parser.add_argument('--name', type = str, required = True, help = 'Name of the experiment to load')

            parser.add_argument('--initial_suffix', type = str, default = 'latest', help = 'Intital suffix')
            parser.add_argument('--batch_size', type = int, default = 32, help = 'Batch size of testing')
            parser.add_argument('--no_load', action = 'store_true', help = 'Specify if not to use stored computations')
            parser.add_argument('--save_suffix', type = str, default = '', help = 'Suffix to save results')

            self.opt = parser.parse_args()

            self.opt.save_result_path = osp.join(osp.join("./checkpoints", self.opt.name), "testing_results")
            if not osp.exists(self.opt.save_result_path):
                os.makedirs(self.opt.save_result_path)

            self.opt.load_weights_path = osp.join("./checkpoints", self.opt.name)

            self.opt.only_backbone = True

            self.opt.device = torch.device(self.opt.gpu_id) if (torch.cuda.is_available() and self.opt.gpu_id >= 0) else torch.device("cpu")


            
    def parse(self):
        return self.opt