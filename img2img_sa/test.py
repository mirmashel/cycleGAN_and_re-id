"""
Adapted from https://github.com/NVlabs/MUNIT
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config, pytorch03_to_pytorch04
from trainer import MUNIT_Trainer, UNIT_Trainer
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="net configuration")
parser.add_argument('--input_folder', type=str, help="input image path")
parser.add_argument('--output_folder', type=str, help="output image path")
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and others for b2a")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--num_style',type=int, default=10, help="number of styles to sample")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
opts = parser.parse_args()



torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

# Load experiment setting
config = get_config(opts.config)

# Setup model and data loader
config['vgg_model_path'] = opts.output_path
if opts.trainer == 'MUNIT':
    style_dim = config['gen']['style_dim']
    trainer = MUNIT_Trainer(config)
elif opts.trainer == 'UNIT':
    trainer = UNIT_Trainer(config)
else:
    sys.exit("Only support MUNIT|UNIT")

try:
    state_dict = torch.load(opts.checkpoint)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])
except:
    state_dict = pytorch03_to_pytorch04(torch.load(opts.checkpoint), opts.trainer)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])

trainer.cuda()
trainer.eval()
encode = trainer.gen_a.encode if opts.a2b else trainer.gen_b.encode # encode function
style_encode = trainer.gen_b.encode if opts.a2b else trainer.gen_a.encode # encode function
decode = trainer.gen_b.decode if opts.a2b else trainer.gen_a.decode # decode function


with torch.no_grad():
    transform = transforms.Compose([transforms.Resize((height, width)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    for im_name in sorted(os.listdir(opts.input_folder)):
        im_path = os.path.join(im_name, opts.input_folder)
        image = Variable(transform(Image.open(im_path).convert('RGB')).unsqueeze(0).cuda())

        content, _ = encode(image)

        if opts.trainer == 'MUNIT':
            style = Variable(torch.randn(1, style_dim, 1, 1).unsqueeze(0).cuda())
    
            outputs = decode(content, s)
            outputs = (outputs + 1) / 2.

            path = os.path.join(opts.output_folder, img_name)
            vutils.save_image(outputs.data, path, padding=0, normalize=True)
        elif opts.trainer == 'UNIT':
            outputs = decode(content)
            outputs = (outputs + 1) / 2.
            path = os.path.join(opts.output_folder, img_name)
            vutils.save_image(outputs.data, path, padding=0, normalize=True)
        else:
            pass

