# *****************
# Model for DCGAN
# *****************

import os
import random
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, generator_dim):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # input is latent vector z 
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels=generator_dim*8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(generator_dim*8),
            nn.ReLU(True),
            # (generator_dim*8) x 4 x 4 - 18
            nn.ConvTranspose2d(in_channels=generator_dim * 8, out_channels=generator_dim * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(generator_dim * 4),
            nn.ReLU(True),
            # (generator_dim*4) x 8 x 8
            nn.ConvTranspose2d(in_channels=generator_dim * 4, out_channels=generator_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(generator_dim * 2),
            nn.ReLU(True),
            # (generator_dim*2) x 16 x 16
            nn.ConvTranspose2d(in_channels=generator_dim * 2, out_channels=generator_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(generator_dim),
            nn.ReLU(True),
            # 4th layer
            nn.ConvTranspose2d(in_channels=generator_dim, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        output = self.main(z)
        return output


class Discriminator(nn.Module):
    def __init__(self, discriminator_dim):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # input is 3 x 64 x 64
            nn.Conv2d(in_channels=3, out_channels=discriminator_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (discriminator_dim) x 32 x 32
            nn.Conv2d(in_channels=discriminator_dim, out_channels=discriminator_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(discriminator_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (discriminator_dim*2) x 16 x 16
            nn.Conv2d(in_channels=discriminator_dim * 2, out_channels=discriminator_dim * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(discriminator_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (discriminator_dim*4) x 8 x 8
            # nn.Conv2d(in_channels=discriminator_dim * 4, out_channels=discriminator_dim * 8, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(discriminator_dim * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            # (discriminator_dim*8) x 4 x 4
            nn.Conv2d(in_channels=discriminator_dim*4, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)

        return output.view(-1, 1).squeeze(1)