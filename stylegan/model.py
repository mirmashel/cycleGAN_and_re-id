import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function
from torchvision import models
import os
from collections import OrderedDict
from math import sqrt

import random


def init_linear(linear):
    init.xavier_normal(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    init.kaiming_normal(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class FusedUpsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = torch.randn(in_channel, out_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = F.conv_transpose2d(input, weight, self.bias, stride=2, padding=self.pad)

        return out


class FusedDownsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = F.conv2d(input, weight, self.bias, stride=2, padding=self.pad)

        return out


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class BlurFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        grad_input = F.conv2d(
            grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
        )

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = F.conv2d(
            gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1]
        )

        return grad_input, None, None


class BlurFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        output = F.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = BlurFunctionBackward.apply(grad_output, kernel, kernel_flip)

        return grad_input, None, None


blur = BlurFunction.apply


class Blur(nn.Module):
    def __init__(self, channel):
        super().__init__()

        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = torch.flip(weight, [2, 3])

        self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
        self.register_buffer('weight_flip', weight_flip.repeat(channel, 1, 1, 1))

    def forward(self, input):
        return blur(input, self.weight, self.weight_flip)
        # return F.conv2d(input, self.weight, padding=1, groups=input.shape[1])


class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        padding,
        kernel_size2=None,
        padding2=None,
        downsample=False,
        fused=False,
    ):
        super().__init__()

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        self.conv1 = nn.Sequential(
            EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
            nn.LeakyReLU(0.2),
        )

        if downsample:
            if fused:
                self.conv2 = nn.Sequential(
                    Blur(out_channel),
                    FusedDownsample(out_channel, out_channel, kernel2, padding=pad2),
                    nn.LeakyReLU(0.2),
                )

            else:
                self.conv2 = nn.Sequential(
                    Blur(out_channel),
                    EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                    nn.AvgPool2d(2),
                    nn.LeakyReLU(0.2),
                )

        else:
            self.conv2 = nn.Sequential(
                EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                nn.LeakyReLU(0.2),
            )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        return out


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out


class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image, noise):
        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        style_dim=512,
        initial=False,
        upsample=False,
        fused=False,
    ):
        super().__init__()

        if initial:
            self.conv1 = ConstantInput(in_channel)

        else:
            if upsample:
                if fused:
                    self.conv1 = nn.Sequential(
                        FusedUpsample(
                            in_channel, out_channel, kernel_size, padding=padding
                        ),
                        Blur(out_channel),
                    )

                else:
                    self.conv1 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        EqualConv2d(
                            in_channel, out_channel, kernel_size, padding=padding
                        ),
                        Blur(out_channel),
                    )

            else:
                self.conv1 = EqualConv2d(
                    in_channel, out_channel, kernel_size, padding=padding
                )

        self.noise1 = equal_lr(NoiseInjection(out_channel))
        self.adain1 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = EqualConv2d(out_channel, out_channel, kernel_size, padding=padding)
        self.noise2 = equal_lr(NoiseInjection(out_channel))
        self.adain2 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu2 = nn.LeakyReLU(0.2)

    def forward(self, input, style, noise): # style.size = (bs, 512), noise.size = (bs, 1, h, w)
        out = self.conv1(input)
        # print(style.size(), noise.size())
        out = self.noise1(out, noise)
        out = self.lrelu1(out)
        out = self.adain1(out, style)

        out = self.conv2(out)
        out = self.noise2(out, noise)
        out = self.lrelu2(out)
        out = self.adain2(out, style)

        return out

    # for inversion in W+ space
    def forward2(self, input, style, noise, adain = True): # style.size = (bs, 2, 512), noise.size = (bs, 1, h, w)
        out = self.conv1(input)
        # out = self.noise1(out, noise)
        out = self.lrelu1(out)
        if adain:
            out = self.adain1(out, style[:,0,:])

        out = self.conv2(out)
        # out = self.noise2(out, noise)
        out = self.lrelu2(out)
        if adain:
            out = self.adain2(out, style[:,1,:])

        return out



class Generator(nn.Module):
    def __init__(self, code_dim, fused=True):
        super().__init__()

        self.progression = nn.ModuleList(
            [
                StyledConvBlock(512, 512, 3, 1, initial=True, style_dim=code_dim),  # 4
                StyledConvBlock(512, 512, 3, 1, upsample=True, style_dim=code_dim),  # 8
                StyledConvBlock(512, 512, 3, 1, upsample=True, style_dim=code_dim),  # 16
                StyledConvBlock(512, 512, 3, 1, upsample=True, style_dim=code_dim),  # 32
                StyledConvBlock(512, 256, 3, 1, upsample=True, style_dim=code_dim),  # 64
                StyledConvBlock(256, 128, 3, 1, upsample=True, style_dim=code_dim, fused=fused),  # 128
                StyledConvBlock(128, 64, 3, 1, upsample=True, style_dim=code_dim, fused=fused),  # 256
                # StyledConvBlock(64, 32, 3, 1, upsample=True, fused=fused),  # 512
                # StyledConvBlock(32, 16, 3, 1, upsample=True, fused=fused),  # 1024
            ]
        )

        self.to_rgb = nn.ModuleList(
            [
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(256, 3, 1),
                EqualConv2d(128, 3, 1),
                EqualConv2d(64, 3, 1),
                # EqualConv2d(32, 3, 1),
                # EqualConv2d(16, 3, 1),
            ]
        )

        # self.blur = Blur()

    def forward(self, style, noise, step=0, alpha=-1, mixing_range=(-1, -1)):
        out = noise[0]

        if len(style) < 2:
            inject_index = [len(self.progression) + 1]

        else:
            inject_index = sorted(random.sample(list(range(step)), len(style) - 1))

        crossover = 0

        for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):
            if mixing_range == (-1, -1):
                if crossover < len(inject_index) and i > inject_index[crossover]:
                    crossover = min(crossover + 1, len(style))

                style_step = style[crossover]

            else:
                if mixing_range[0] <= i <= mixing_range[1]:
                    style_step = style[1]

                else:
                    style_step = style[0]

            if i > 0 and step > 0:
                out_prev = out
                
            out = conv(out, style_step, noise[i])

            if i == step:
                out = to_rgb(out)

                if i > 0 and 0 <= alpha < 1:
                    skip_rgb = self.to_rgb[i - 1](out_prev)
                    skip_rgb = F.interpolate(skip_rgb, scale_factor=2, mode='nearest')
                    out = (1 - alpha) * skip_rgb + alpha * out

                break

        return out

    def forward2(self, style, noise, step=0, alpha=-1, mixing_range=(-1, -1), adain_count = 14): # style.size = (bs, 14, 512)
        # print(len(style), style[0].size())
        out = noise[0]

        if len(style) < 2:
            inject_index = [len(self.progression) + 1]

        else:
            inject_index = sorted(random.sample(list(range(step)), len(style) - 1))

        crossover = 0

        for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):
            if mixing_range == (-1, -1):
                if crossover < len(inject_index) and i > inject_index[crossover]:
                    crossover = min(crossover + 1, len(style))

                style_step = style[crossover][:, i*2:i*2+2, :]

            else:
                if mixing_range[0] <= i <= mixing_range[1]:
                    style_step = style[1]

                else:
                    style_step = style[0]

            if i > 0 and step > 0:
                out_prev = out

            out = conv.forward2(out, style_step, noise[i], (i + 1) * 2 <= adain_count)

            if i == step:
                out = to_rgb(out)

                if i > 0 and 0 <= alpha < 1:
                    skip_rgb = self.to_rgb[i - 1](out_prev)
                    skip_rgb = F.interpolate(skip_rgb, scale_factor=2, mode='nearest')
                    out = (1 - alpha) * skip_rgb + alpha * out

                break

        return out

class OrigStyledGenerator(nn.Module):
    def __init__(self, code_dim=512, n_mlp=8):
        super().__init__()

        self.generator = Generator(code_dim)

        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(code_dim, code_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.style = nn.Sequential(*layers)

    def forward(
        self,
        input,
        noise=None,
        step=0,
        alpha=-1,
        mean_style=None,
        style_weight=0,
        mixing_range=(-1, -1),
    ):
        styles = []
        if type(input) not in (list, tuple):
            input = [input]

        for i in input:
            styles.append(self.style(i))

        batch = input[0].shape[0]

        if noise is None:
            noise = []

            for i in range(step + 1):
                size = 4 * 2 ** i
                noise.append(torch.randn(batch, 1, size, size, device=input[0].device))

        if mean_style is not None:
            styles_norm = []

            for style in styles:
                styles_norm.append(mean_style + style_weight * (style - mean_style))

            styles = styles_norm

        return self.generator(styles, noise, step, alpha, mixing_range=mixing_range)

    def mean_style(self, input):
        style = self.style(input).mean(0, keepdim=True)
        return style

    def forward2(
        self,
        input,
        noise=None,
        step=0,
        alpha=-1,
        mean_style=None,
        style_weight=0,
        mixing_range=(-1, -1),
    ):
        styles = [input]
        input = [input]
        # print(input[0].size())

        # if type(input) not in (list, tuple):
        #     input = [input]

        # for i in input:
        #     styles.append(self.style(i))

        batch = input[0].shape[0]

        if noise is None:
            noise = []

            for i in range(step + 1):
                size = 4 * 2 ** i
                noise.append(torch.randn(batch, 1, size, size, device=input[0].device))

        if mean_style is not None:
            styles_norm = []

            for style in styles:
                styles_norm.append(mean_style + style_weight * (style - mean_style))

            styles = styles_norm

        return self.generator.forward2(styles, noise, step, alpha, mixing_range=mixing_range)

        

class StyledGenerator(nn.Module):
    def __init__(self, code_dim=512, n_mlp=None, classes = 700, use_cls = True, static_noise = False, active_style_layers = 14, decoder = 'base', code_first = False, use_face_weights = False):
        super().__init__()

        self.generator = Generator(code_dim)

        self.use_mlp = False
        if n_mlp is not None:
            self.use_mlp = True
            layers = [PixelNorm()]
            for i in range(n_mlp):
                layers.append(EqualLinear(code_dim, code_dim))
                layers.append(nn.LeakyReLU(0.2))

            self.style = nn.Sequential(*layers)

        if use_face_weights:
            ckpt = torch.load("./stylegan-256px-new.model")

            old_ckpt_g = ckpt['g_running']
            new_ckpt_g = OrderedDict()
            for k in old_ckpt_g:
                splited = k.split('.')
                if (splited[0] == 'generator' and (splited[1] == 'progression' or splited[1] == 'to_rgb') and int(splited[2]) <= 6):
                    new_ckpt_g.update({'.'.join(splited[1:]): old_ckpt_g[k]})
            self.generator.load_state_dict(new_ckpt_g)

            new_ckpt_g = OrderedDict()
            for k in old_ckpt_g:
                splited = k.split('.')
                if splited[0] == 'style':
                    new_ckpt_g.update({'.'.join(splited[1:]): old_ckpt_g[k]})
            self.style.load_state_dict(new_ckpt_g)


        if decoder == 'base':
            self.synt_img_style = nn.Sequential(
                EqualConv2d(3, 32, 3, padding=1),
                nn.LeakyReLU(0.2),
                EqualConv2d(32, 64, 4, padding=1),
                nn.MaxPool2d(2, stride=2),
                nn.LeakyReLU(0.2),
                EqualConv2d(64, 128, 4, padding=1),
                nn.MaxPool2d(2, stride=2),
                nn.LeakyReLU(0.2),
                EqualConv2d(128, 256, 3, padding=1),
                nn.MaxPool2d(2, stride=2),
                nn.LeakyReLU(0.2),
                EqualConv2d(256, 512, 4, padding=1),
                nn.LeakyReLU(0.2),
                EqualConv2d(512, code_dim, 4, padding=1),
                nn.AdaptiveMaxPool2d((1,1)),
                nn.Flatten(),
                nn.LeakyReLU(0.2),
                EqualLinear(code_dim, code_dim),
                nn.LeakyReLU(0.2),
                EqualLinear(code_dim, code_dim),
            )
        elif decoder == 'resnet50':
            self.synt_img_style = models.resnet50(pretrained = True)
            self.synt_img_style.fc = nn.Linear(2048, code_dim)
            # self.synt_img_style = nn.Sequential(*(list(self.synt_img_style.children())[:-1]), nn.Flatten(), nn.Linear(2048, code_dim))

        elif decoder == 'resnet34':
            self.synt_img_style = models.resnet34(pretrained = True)
            self.synt_img_style.fc = nn.Linear(512, code_dim)
            # self.synt_img_style = nn.Sequential(*(list(self.synt_img_style.children())[:-1]), nn.Flatten(), nn.Linear(512, code_dim))




        self.use_cls = classes is not None and use_cls

        if self.use_cls:
            self.classifier_on_style = nn.Sequential(
                nn.LeakyReLU(0.2),
                EqualLinear(code_dim, code_dim),
                nn.LeakyReLU(0.2),
                EqualLinear(code_dim, classes),
            )
        else:
            self.classifier_on_style = None

        if code_first:
            self.init_noise_16 = nn.Sequential(EqualLinear(code_dim, 16))
            self.init_noise_64 = nn.Sequential(EqualLinear(code_dim, 64))

        else:
            self.init_noise = nn.Sequential(EqualLinear(code_dim, 16))

        self.active_style_layers = active_style_layers



        self.static_noise = static_noise

        self.code_first = code_first


    def forward(
        self,
        img_enc,
        noise=None,
        step=0,
        alpha=-1,
    ):
        styles = []

        img_code = self.synt_img_style(img_enc)

        if self.use_mlp:
            styles.append(self.style(img_code))
        else:
            styles.append(img_code)

        batch = img_code.shape[0]

        if noise is None:
            noise = []

            for i in range(step + 1):
                size = 4 * 2 ** i
                if self.static_noise:
                    noise.append(torch.zeros(batch, 1, size, size, device=img_code.device))
                else:
                    noise.append(torch.randn(batch, 1, size, size, device=img_code.device))
            if self.code_first:
                noise[0] = self.init_noise_16(img_code).view(-1, 1, 4, 4)
                noise[1] = self.init_noise_64(img_code).view(-1, 1, 8, 8)

        if self.classifier_on_style is not None:
            classifier = self.classifier_on_style(img_code)
        else:
            classifier = None

        if self.active_style_layers == 14:
            generator_output = self.generator(styles, noise, step, alpha)
        else:
            styles[0] = torch.cat([styles[0].unsqueeze(1) for i in range(14)], 1)
            generator_output = self.generator.forward2(styles, noise, step, alpha, adain_count = self.active_style_layers)

        return generator_output, classifier

    def mean_style(self, input):
        style = self.style(input).mean(0, keepdim=True)

        return style


class Discriminator(nn.Module):
    def __init__(self, fused=True, from_rgb_activate=False, use_face_weights = False):
        super().__init__()

        self.progression = nn.ModuleList(
            [
                # ConvBlock(16, 32, 3, 1, downsample=True, fused=fused),  # 512
                ConvBlock(32, 64, 3, 1, downsample=True, fused=fused),  # 256
                ConvBlock(64, 128, 3, 1, downsample=True, fused=fused),  # 128
                ConvBlock(128, 256, 3, 1, downsample=True, fused=fused),  # 64
                ConvBlock(256, 512, 3, 1, downsample=True),  # 32
                ConvBlock(512, 512, 3, 1, downsample=True),  # 16
                ConvBlock(512, 512, 3, 1, downsample=True),  # 8
                ConvBlock(512, 512, 3, 1, downsample=True),  # 4
                ConvBlock(513, 512, 3, 1, 4, 0),
            ]
        )

        def make_from_rgb(out_channel):
            if from_rgb_activate:
                return nn.Sequential(EqualConv2d(3, out_channel, 1), nn.LeakyReLU(0.2))

            else:
                return EqualConv2d(3, out_channel, 1)

        self.from_rgb = nn.ModuleList(
            [
                # make_from_rgb(16),
                make_from_rgb(32),
                make_from_rgb(64),
                make_from_rgb(128),
                make_from_rgb(256),
                make_from_rgb(512),
                make_from_rgb(512),
                make_from_rgb(512),
                make_from_rgb(512),
            ]
        )

        # self.blur = Blur()

        self.n_layer = len(self.progression)

        self.linear = EqualLinear(512, 1)

        if use_face_weights:
            ckpt = torch.load("./stylegan-256px-new.model")
            old_ckpt_disc = ckpt['discriminator']
            new_ckpt_disc = OrderedDict()
            for k in old_ckpt_disc:
                splited = k.split('.')
                if (splited[0] == 'progression' or splited[0] == 'from_rgb') and int(splited[1]) > 0:
                    splited[1] = str(int(splited[1]) - 1)
                    # if splited[0] == 'from_rgb':
                    #     splited.pop(2)
                    new_k = '.'.join(splited)
                    new_ckpt_disc.update({new_k: old_ckpt_disc[k]})
                if splited[0] == 'linear':
                    new_ckpt_disc.update({k: old_ckpt_disc[k]})

            self.load_state_dict(new_ckpt_disc)

    def forward(self, input, step=0, alpha=-1):
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb[index](input)

            if i == 0:
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 0:
                if i == step and 0 <= alpha < 1:
                    skip_rgb = F.avg_pool2d(input, 2)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)

                    out = (1 - alpha) * skip_rgb + alpha * out

        out = out.squeeze(2).squeeze(2)
        # print(input.size(), out.size(), step)
        out = self.linear(out)

        return out




class VGG(nn.Module):
    def __init__(self, resolution = 8, num_classes = 700, only_features = False):
        super(VGG, self).__init__()
        
        self.only_features = only_features
        self.resolution = resolution
        self.vgg13 = models.vgg13_bn(pretrained = True) # False на True

        if not only_features:
            self.vgg13.avgpool = nn.AdaptiveAvgPool2d((2, 2))
            self.vgg13.classifier = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(True),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Linear(512, num_classes),
            )

        if resolution == 8:
            self.vgg13.features = self.vgg13.features[:-8]
            
        if resolution == 16:
            self.vgg13.features = self.vgg13.features[:-1]
        
    def forward(self, x):
        return self.vgg13(x)
        
    def save(self, save_path, save_name = 'VGG_13'):
        os.makedirs(save_path, exist_ok=True)
        save_name = save_name + '_' + str(self.resolution) + '.pth'
        save_name = os.path.join(save_path, save_name)

        if self.only_features:
            param_to_save = {
                'features' : self.vgg13.features.state_dict(),
            }
        else:
            param_to_save = {
                'features' : self.vgg13.features.state_dict(),
                'avgpool' : self.vgg13.avgpool.state_dict(),
                'classifier' : self.vgg13.classifier.state_dict(),
            }

        torch.save(param_to_save, save_name)
        
    def load(self, load_path, load_name = 'VGG_13'):
        load_name = load_name + '_' + str(self.resolution) + '.pth'
        load_name = os.path.join(load_path, load_name)
        ckpt = torch.load(load_name, map_location = 'cpu')
        
        self.vgg13.features.load_state_dict(ckpt['features'])
        if not self.only_features:
            self.vgg13.avgpool.load_state_dict(ckpt['avgpool']) 
            self.vgg13.classifier.load_state_dict(ckpt['classifier']) 

        # self.vgg13.load_state_dict(ckpt['vgg'])



class PerceptualLoss_v1(nn.Module):
    def __init__(self, weights = [1.0, 1.0, 1.0, 1.0, 1.0], resolution = 8, load_path = '', load_prefix = "imagenet"):
        super(PerceptualLoss_v1, self).__init__()
        self.vgg = VGG(resolution, only_features = True)
        if load_prefix != "imagenet":
            self.vgg.load(load_path, load_prefix)
        self.vgg.eval()
        self.weights = weights
        self.criterion = torch.nn.L1Loss()

    def __call__(self, x, y, synt_imgs_loss=False):
        # Compute features
        x_vgg1, x_vgg2, x_vgg3, x_vgg4 = self.get_img_features(x)
        y_vgg1, y_vgg2, y_vgg3, y_vgg4 = self.get_img_features(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg1, y_vgg1)
        content_loss += self.weights[1] * self.criterion(x_vgg2, y_vgg2)
        content_loss += self.weights[2] * self.criterion(x_vgg3, y_vgg3)
        content_loss += self.weights[3] * self.criterion(x_vgg4, y_vgg4)
        return content_loss

    def get_img_features(self, x):
        x_vgg1 = self.vgg.vgg13.features[:4](x)
        x_vgg2 = self.vgg.vgg13.features[4:8](x_vgg1)
        x_vgg3 = self.vgg.vgg13.features[8:15](x_vgg2)
        x_vgg4 = self.vgg.vgg13.features[15:22](x_vgg3)
        return x_vgg1, x_vgg2, x_vgg3, x_vgg4
