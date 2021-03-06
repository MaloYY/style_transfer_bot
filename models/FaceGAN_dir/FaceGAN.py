import os
import torch
import torch.nn as nn
from torchvision.utils import save_image


class FaceGAN:
    def __init__(self):
        if not os.path.exists(f'models/FaceGAN_dir/faces'):
            os.makedirs('models/FaceGAN_dir/faces')
        self.device = torch.device('cpu')
        self.latent_size = 512
        self.generator = nn.Sequential(

            # in: latent_size x 1 x 1
            nn.ConvTranspose2d(self.latent_size, 1024, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 1024 x 4 x 4

            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 512 x 16 x 16

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 256 x 32 x 32

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 128 x 64 x 64

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 128 x 128 x 128

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.Tanh()
            nn.Sigmoid()
            # out: 3 x 128 x 128
        )

        self.model = self.generator.to(device=self.device)
        self.model.load_state_dict(torch.load('models/model_weights/genr3m128p10e.model', map_location=self.device))
        self.model.eval()

    async def get_image(self):
        """Generating a random tensor,
         passing through the model and saving the picture"""

        # generating
        fixed_latent = torch.randn(1, 512, 1, 1, device=self.device)

        with torch.no_grad():
            # passing through
            fake_images = self.model(fixed_latent)
            # saving
            save_image(fake_images, f'models/FaceGAN_dir/faces/fake.jpg')


    async def remove_image(self):
        if os.path.isfile(f'models/FaceGAN_dir/faces/fake.jpg'):
            os.remove(f'models/FaceGAN_dir/faces/fake.jpg')


if __name__ == '__main__':
    inst = FaceGAN()
    #inst.get_image()
    inst.remove_image()
