import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod


class BaseDataset(data.Dataset, ABC):
    def __init__(self, config):
        self.config = config
        self.root = config.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __getitem__(self, index):
        pass


def get_params(config, size):
    w, h = size
    new_h = h
    new_w = w
    if config.preprocess == 'resize_and_crop':
        new_h = new_w = config.load_size
    elif config.preprocess == 'scale_width_and_crop':
        new_w = config.load_size
        new_h = config.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - config.crop_size))
    y = random.randint(0, np.maximum(0, new_h - config.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(config, params=None, grayscale=False, method=transforms.InterpolationMode.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in config.preprocess:
        osize = [config.load_size, config.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in config.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, config.load_size, config.crop_size, method)))

    if 'crop' in config.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(config.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], config.crop_size)))

    if config.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not config.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __transforms2pil_resize(method):
    mapper = {transforms.InterpolationMode.BILINEAR: Image.BILINEAR,
              transforms.InterpolationMode.BICUBIC: Image.BICUBIC,
              transforms.InterpolationMode.NEAREST: Image.NEAREST,
              transforms.InterpolationMode.LANCZOS: Image.LANCZOS,}
    return mapper[method]


def __make_power_2(img, base, method=transforms.InterpolationMode.BICUBIC):
    method = __transforms2pil_resize(method)
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_size, crop_size, method=transforms.InterpolationMode.BICUBIC):
    method = __transforms2pil_resize(method)
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
