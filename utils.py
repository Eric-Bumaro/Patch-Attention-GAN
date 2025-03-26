import os

import numpy as np
import torch
from PIL import Image


def print_options(config):
    message = ''
    message += '----------------- Options ---------------\n'
    values = vars(config)
    for k, v in sorted(values.items()):
        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    save_suffix = os.path.join(config.result_dir, config.experiment_name)
    os.makedirs(save_suffix, exist_ok=True)
    with open(os.path.join(save_suffix, "setting.txt"), "w") as file:
        file.write(message)
    print(message)


def tensor2im(input_image, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def set_gpu_id(config):
    torch.cuda.set_device(config.gpu_ids[0])


def print_networks(nets):
    print('---------- Networks initialized -------------')
    for net in nets:
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print('Total number of parameters : %.3f M' % (num_params / 1e6))
    print('-----------------------------------------------')


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def save_images(config, visuals, image_path, aspect_ratio=1.0):
    image_dir = os.path.join(config.result_dir, config.experiment_name)
    os.makedirs(image_dir, exist_ok=True)
    base_name = os.path.basename(image_path[0])
    name = base_name[:base_name.index(".")]
    for label, im_data in visuals.items():
        im = tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        save_image(im, save_path, aspect_ratio=aspect_ratio)
