from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

import copy
import os

from losses import Normalization, ContentLoss, StyleLoss
import asyncio

device = torch.device("cpu")


def image_loader(image_name, imsize):
    loader = transforms.Compose([
        transforms.Resize(imsize),  # нормируем размер изображения
        transforms.CenterCrop(imsize),
        transforms.ToTensor()])  # превращаем в удобный формат

    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img, content_layers, style_layers):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            # Переопределим relu уровень
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    # выбрасываем все уровни после последенего styel loss или content loss
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    #model.load_state_dict(torch.load('model_weights/EasyST.model'))
    #model = torch.load('model_weights/EasyST1.model')
    #model.eval()

    return model, style_losses, content_losses


def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    # добоваляет содержимое тензора катринки в список изменяемых оптимизатором параметров
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, content_layers, style_layers,
                       num_steps=500, style_weight=100000, content_weight=1):
    """Run the style transfer."""
    # print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                     normalization_mean, normalization_std,
                                                                     style_img, content_img,
                                                                     content_layers, style_layers)
    optimizer = get_input_optimizer(input_img)

    # print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values
            # это для того, чтобы значения тензора картинки не выходили за пределы [0;1]
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()

            model(input_img)

            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            # взвешивание ошибки
            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img


class StyleTransfer:
    def __init__(self, content_path, style_path, user_id):
        self.user_id = user_id
        self.imsize = 128

        self.content_path = content_path
        self.style_path = style_path
        self.trans_path = f'transferred/image{str(self.user_id)}.jpg'

        self.content_img = image_loader(content_path, self.imsize)  # as well as here
        self.style_img = image_loader(style_path, self.imsize)  # измените путь на тот который у вас.
        self.input_img = self.content_img.clone()

        self.content_layers_default = ['conv_4']
        self.style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    async def transfer(self):
        cnn = models.vgg19(pretrained=False).features.to(device).eval()
        cnn.load_state_dict(torch.load('model_weights/vgg19_features.model'))
        output = run_style_transfer(cnn, self.mean, self.std, self.content_img, self.style_img,
                                    self.input_img, self.content_layers_default, self.style_layers_default)
        save_image(output, self.trans_path)

    async def clear(self):
        if os.path.isfile(self.trans_path):
            os.remove(self.trans_path)

        if os.path.isfile(self.content_path):
            os.remove(self.content_path)

        if os.path.isfile(self.style_path):
            os.remove(self.style_path)


if __name__ == '__main__':
    inst = StyleTransfer(f'content/cnt58369298.jpg', f'style/stl58369298.jpg', 58369298)
    asyncio.get_event_loop().run_until_complete( inst.transfer() )
