from PIL import Image

import functools
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import os

from models.CycleGAN_dir.ResNet import ResnetGenerator

device = torch.device("cpu")

WEIGHTS = 'models/model_weights/G_A4.374477.model'


def _patch_instance_norm_state_dict(state_dict, module, keys, i=0):
    """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
    key = keys[i]
    if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
        if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'running_mean' or key == 'running_var'):
            if getattr(module, key) is None:
                state_dict.pop('.'.join(keys))
        if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'num_batches_tracked'):
            state_dict.pop('.'.join(keys))
    else:
        _patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
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


def image_loader(image_name, imsize):
    loader = transforms.Compose([
        transforms.Resize(imsize),  # нормируем размер изображения
        transforms.CenterCrop(imsize),
        transforms.ToTensor()])  # превращаем в удобный формат

    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


class Summer2Winter:
    def __init__(self, real_path, fake_path):
        self.imsize = 256

        self.real_path = real_path
        self.fake_path = fake_path

        # create model
        self.norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        self.GenA = ResnetGenerator(3, 3, 64, norm_layer=self.norm_layer, use_dropout=False, n_blocks=9)
        self.load_networks()

    def load_networks(self):
        state_dict = torch.load(WEIGHTS, map_location=torch.device('cpu'))

        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        # patch InstanceNorm checkpoints prior to 0.4
        for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
            _patch_instance_norm_state_dict(state_dict, self.GenA, key.split('.'), i=0)
        self.GenA.load_state_dict(state_dict)

    async def get_image(self):
        # load image
        real_img = image_loader(self.real_path, self.imsize)
        # transfer
        fake_img = self.GenA(real_img)
        # preparing to output
        result_img = tensor2im(fake_img)
        # output
        im = Image.fromarray(result_img)
        im.save(self.fake_path)

    async def clear(self):
        if os.path.isfile(self.real_path):
            os.remove(self.real_path)
        if os.path.isfile(self.fake_path):
            os.remove(self.fake_path)


if __name__ == '__main__':
    inst = Summer2Winter('models/CycleGAN_dir/real/kartinki24_ru_summer_124.jpg', 'models/CycleGAN_dir/fake/kartinki24_ru_summer_124.jpg')
    # use it for weights: WEIGHTS = '../model_weights/G_A4.374477.model'
    # asyncio.run(inst.clear())
