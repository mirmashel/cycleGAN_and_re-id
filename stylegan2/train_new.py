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
from network.model import Embedder, Generator, Discriminator
from dataset import AlignedDatasetLoader
import os
import time
import matplotlib
import skimage.io as io


def save_resized_images(save_name, images):
    num_cols = 7
    num_rows = 3
    target_shape = (120, 50)
    result_image = np.zeros((num_rows * target_shape[0], num_cols * target_shape[1], 3), dtype = np.float32)
    for nr in range(num_rows):
        for nc in range(num_cols):
            im = torch.clamp(images[nr * num_cols + nc][0], -1, 1).numpy()
            im = (np.transpose(im, (1, 2, 0)) + 1) / 2.0 * 255.0
            im = sk_transform.resize(im, target_shape)
            result_image[target_shape[0] * nr:target_shape[0] * (nr + 1), target_shape[1] * nc:target_shape[1] * (nc + 1), :] = im[:, :, :]

    io.imsave(save_name, result_image.astype(np.uint8))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Unet adaptiveInstance')

    parser.add_argument('--name', type=str, required = True, help='Name of experiment')

    parser.add_argument('--source_path', required = True, type = str)
    parser.add_argument('--target_path', required = True, type = str)
    parser.add_argument('--sample_iters', type = int, default = 1000)
    parser.add_argument('--save_epochs', type = int, default = 1)

    parser.add_argument('--num_epochs', type=int, default=100, help='number of samples used for each training phases')
    parser.add_argument('--batch_size', type=int, default=4, help='number of samples used for each training phases')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--pic_size', default=224, type=int, help='max image size')
    parser.add_argument('--pad_size_to', default = 256, type = int)
    parser.add_argument('--code_size', default=512, type=int)
    parser.add_argument('--n_mlp', default=8, type=int)

    parser.add_argument('--lambda_l1', default = 1., type = float)
    parser.add_argument('--lambda_cnt', default = 1., type = float)
    pasrer.add_argument('--gan_mode', type = str, default = 'orig', help = '[orig | lsgan]')

    parser.add_argument('--ckpt_name', default=None, type=str, help='load from previous checkpoints')

    args = parser.parse_args()
    args.experiment_dir = os.path.join("./checkpoints", args.name)
    os.makedirs(args.experiment_dir, exist_ok = True)

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = AlignedDatasetLoader(args.source_path, args.target_path, transform, resolution = args.pic_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last = True)

    G = nn.DataParallel(Generator(args.pic_size, args.pad_size_to, args.code_size, args.n_mlp).cuda())
    if args.gan_mode == 'orig':
        D = nn.DataParallel(Discriminator(args.pic_size, args.pad_size_to).cuda())
    else if args.gan_mode == 'lsgan':
        D = nn.DataParallel(Discriminator(3).cuda())

    optimizerG = optim.Adam(params = G.parameters(),
                        lr=5e-5,
                        amsgrad=False)

    optimizerD = optim.Adam(params = D.parameters(),
                            lr=2e-4,
                            amsgrad=False)

    criterionG = LossG(device='cuda', args.lambda_l1, args.lambda_cnt, args.gan_mode)
    criterionDreal = LossDSCreal(args.gan_mode)
    criterionDfake = LossDSCfake(args.gan_mode)

    epoch = 0
    total_samples = 0
    start_epoch = 0

    if args.ckpt_name is not None:
        # ckpt_name = os.path.join("./checkpoints", args.ckpt_name)
        ckpt = torch.load(args.ckpt_name)

        start_epoch = ckpt['ckpt_epoch']
        total_samples = ckpt['total_samples']
        G.module.load_state_dict(ckpt['G'])
        D.module.load_state_dict(ckpt['D'])
        optimizerG.load_state_dict(ckpt['optimizerG'])
        optimizerD.load_state_dict(ckpt['optimizerD'])


    if args.ckpt_name is None:
        def init_weights(m):
            if type(m) == nn.Conv2d or type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)

        G.apply(init_weights)
        D.apply(init_weights)

        sample_path_name = os.path.join(args.experiment_dir, "sample")
        if os.path.exists(sample_path_name):
            os.rmtree(sample_path_name)

        save_name = os.path.join(args.experiment_dir, f'{str(0).zfill(3)}.model')

        print('Initiating new checkpoint...')
        torch.save({
                'ckpt_epoch': epoch,
                'total_samples': total_samples,
                'G': G.module.state_dict(),
                'D': D.module.state_dict(),
                'optimizerG': optimizerG.state_dict(),
                'optimizerD': optimizerD.state_dict()
                }, save_name)
        print('...Done')

    np.random.seed(10001)
    gen_i, gen_j = (7, 3)
    idxs = np.random.randint(low = 0, high = len(dataset), size = (21, ))
    fixed_batch = [dataset[idx]['img_from'].unsqueeze(0) for idx in idxs]
    save_name = os.path.join(sample_path_name, f'{str(0).zfill(6)}.png')
    save_resized_images(save_name, fixed_batch)
    np.random.seed(time.time())
    matplotlib.use('agg')

    for epoch in range(start_epoch, args.num_epochs):

        pbar = tqdm(dataloader, leave=True, initial=0)
        pbar.set_postfix(epoch=epoch)

        for i_batch, data in enumerate(pbar, start=0):
            img_from = data['img_from'].cuda()
            img_to = data['img_to'].cuda()
            t0 = time.time()

            gen_in = torch.randn(args.batch_size, args.code_size, device='cuda')

            with torch.autograd.enable_grad():
                #zero the parameter gradients
                optimizerG.zero_grad()
                optimizerD.zero_grad()

                #train G and D
                img_from_hat = G(img_from, gen_in) 
                r_hat = D(img_from_hat)
                # with torch.no_grad():
                #     r, D_res_list = D(img_to)

                lossG = criterionG(img_to, img_from, img_from_hat, r_hat)
                
                lossG.backward(retain_graph=False)
                optimizerG.step()
            
            with torch.autograd.enable_grad():
                optimizerG.zero_grad()
                optimizerD.zero_grad()
                img_from_hat.detach_().requires_grad_()
                r_hat = D(img_from_hat)
                lossDfake = criterionDfake(r_hat)

                r = D(img_to)
                lossDreal = criterionDreal(r)
                
                lossD = lossDfake + lossDreal
                lossD.backward(retain_graph=False)
                optimizerD.step()

            total_samples += img_from.size(0)

            if (i_batch + 1) % args.sample_iters == 0:
                images = []
                G.eval()
                with torch.no_grad():
                    for f_im in fixed_batch:
                        gen_in = torch.randn(1, args.batch_size, args.code_size, device='cuda')
                        images.append(
                            G(f_im.cuda(), gen_in).data.cpu()
                        )

                save_name = os.path.join(args.experiment_dir, "sample")
                os.makedirs(save_name, exist_ok = True)
                save_name = os.path.join(save_name, f'{str(total_samples).zfill(6)}.png')

                save_resized_images(save_name, images)

                G.train()

            state_msg = (
                f'Iters: {i + 1}; total_samples: {total_samples}; Iter time: {(time.time() - t0):.2f}'
            )
            
            state_msg += f'; G: {lossG:.3f}; D: {lossD:.3f};'

            pbar.set_description(state_msg)


        if (epoch + 1) % args.save_epochs == 0 or epoch + 1 == args.num_epochs:
            save_name = os.path.join(sample_path_name, f'{str(epoch + 1).zfill(3)}.model')
            print('Initiating new checkpoint...')
            torch.save({
                    'ckpt_epoch': epoch,
                    'total_samples': total_samples,
                    'G': G.module.state_dict(),
                    'D': D.module.state_dict(),
                    'optimizerG': optimizerG.state_dict(),
                    'optimizerD': optimizerD.state_dict()
                    }, save_name)
            print('...Done')