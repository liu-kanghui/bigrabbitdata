'''
Author: Kanghui Liu
Date: 6/7/2020

This is a Pytorch 1.6.0.dev20200526 implementation of 
Generative adversarial networks(GANs) with CelebA dataset
using convolutional neural network

Blog in depth tutorial: https://www.bigrabbitdata.com/

Gan Hack: weight initialization

'''


import torch
import torchvision
import torch.nn as nn
import torch.optim as optim 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import itertools
import matplotlib.pyplot as plt 
import numpy as np

import subprocess
import os


class Discriminator(nn.Module):
    '''
        Discriminator model
    '''
    def __init__(self, image_channels, features):
        super().__init__()

        # input image size : 3 x 64 x 64
        self.conv_0 = nn.Sequential( 
            # Batch_size x image_channels x 64 x 64
            nn.Conv2d(image_channels, features, 
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )

        self.conv_1 = nn.Sequential(
            # Batch_size x features  x 32 x 32
            nn.Conv2d(features, features * 2, 
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features*2),
            nn.LeakyReLU(0.2)
        )

        self.conv_2 = nn.Sequential(
            # Batch_size x (features * 2) x 16 x 16
            nn.Conv2d(features * 2, features * 4, 
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features*4),
            nn.LeakyReLU(0.2)
        )

        self.conv_3 = nn.Sequential(
            # Batch_size x (features * 4) x 8 x 8
            nn.Conv2d(features * 4, features * 8, 
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features*8),
            nn.LeakyReLU(0.2)
        )

        self.out = nn.Sequential(
             # Batch_size x (features * 8) x 4 x 4
            nn.Conv2d(features * 8, 1, 
                      kernel_size=4, stride=2, padding=0),
            # Batch x 1 x 1 x 1
            nn.Sigmoid()
        )


    # custom weights initialization
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


    def forward(self, x):

        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.out(x)
        # reduce the dimension from 128x1x1x1 to 128
        return torch.squeeze(x)


class Generator(nn.Module):
    '''
        Generator model
    '''
    def __init__(self, noise_features, image_channels, features):
        super().__init__()


        self.deconv_0 = nn.Sequential(
            # Batch_size x noise_features x 1 x 1
            nn.ConvTranspose2d(noise_features, features * 8, 
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(features * 8),
            nn.ReLU()
        )

        self.deconv_1 = nn.Sequential(     
            # Batch x  (features * 8) x 4 x 4     
            nn.ConvTranspose2d(features * 8, features * 4, 
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.ReLU()
        )

        self.deconv_2 = nn.Sequential(
            # Batch x (features * 4) x 8 x 8 
            nn.ConvTranspose2d(features * 4, features * 2, 
                               kernel_size=4, stride=2,  padding=1),
            nn.BatchNorm2d(features*2),
            nn.ReLU()
        )

        self.deconv_3 = nn.Sequential(
            # Batch x (features * 2) x 16 x 16 
            nn.ConvTranspose2d(features * 2, features, 
                               kernel_size=4, stride=2,  padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU()
        )

        self.out = nn.Sequential(
            # Batch x (features) x 32 x 32
            nn.ConvTranspose2d(features, image_channels, 
                               kernel_size=4, stride=2, padding=1),
            # Batch x image_channels X 64 X 64
            nn.Tanh()
        )

    # custom weights initialization
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        x = self.deconv_0(x)
        x = self.deconv_1(x)
        x = self.deconv_2(x)
        x = self.deconv_3(x)
        x = self.out(x)
        return x



def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()




noise_features = 100

def noise(data_size):
    '''
    Generates data_size number of random noise
    '''
    n = torch.randn(data_size, noise_features, 1, 1)
    return n


def im_convert(tensor):
    '''
        Convert Tensor to displable format
    '''
    image = tensor.to("cpu").clone().detach()
    image = image.numpy()
    image = image.transpose(1,2,0)
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    image = image.clip(0, 1)

    return image


# Loading training dataset
transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,)),
    ])

dataset = datasets.CelebA(root='dataset/', split='train', 
                        transform=transforms, download=True)
dataloader = DataLoader(dataset, batch_size=128, 
                        drop_last=True,
                        shuffle=True)


# Loading models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create discriminator and generator
discriminator = Discriminator(image_channels=3, 
                               features=20).to(device)


generator = Generator(noise_features, 
                      image_channels=3, 
                      features=20).to(device)


# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
discriminator.weight_init(mean=0.0, std=0.02)
generator.weight_init(mean=0.0, std=0.02)


# Create 100 test_noise for visualizing how well our model perform.
test_noise = noise(100).to(device)


# Optimizers
lr = 0.0002
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()


# True and False Labels.  128 is the batch size
true_label = torch.ones(128).to(device)
false_label = torch.zeros(128).to(device)


# Create folder to hold result
result_folder = 'gan2-1-result'
if not os.path.exists(result_folder ):
    os.makedirs(result_folder )

# Training in action
print("Starting Training...")

num_epochs = 100
discriminator_loss_history = []
generator_loss_history = []

for epoch in range(1, num_epochs+1):
    discriminator_batch_loss = 0.0
    generator_batch_loss = 0.0

    for batch_idx, (data, targets) in enumerate(dataloader):
        data = data.to(device)
        batch_size = data.shape[0]

        discriminator.zero_grad()

        # Train discriminator to get better at differentiate real/fake data
        # 1.1 Train discriminator on real data
        d_real_predict = discriminator(data)
        d_real_loss = criterion(d_real_predict, true_label)
  

        # 1.2 Train discriminator on fake data from generator
        d_fake_noise = noise(batch_size).to(device)
        # Generate outputs and detach to avoid training the Generator on these labels
        d_fake_input = generator(d_fake_noise).detach()
        d_fake_predict = discriminator(d_fake_input)
        d_fake_loss = criterion(d_fake_predict, false_label)

        # 1.3 combine real loss and fake loss for discriminator
        discriminator_loss = d_real_loss + d_fake_loss
        discriminator_batch_loss += discriminator_loss.item()
        discriminator_loss.backward()
        optimizerD.step()


        # Train generator to get better at deceiving discriminator
        g_fake_noise = noise(batch_size).to(device)
        g_fake_input = generator(g_fake_noise)
        generator.zero_grad()
        # Get prediction from discriminator
        g_fake_predict = discriminator(g_fake_input)
        generator_loss = criterion(g_fake_predict, true_label)
        generator_batch_loss += generator_loss.item()
        generator_loss.backward()
        optimizerG.step()

        if (batch_idx + 1) % 100 == 0:

            print(f'Epoch [{epoch}/{num_epochs}]  Batch {batch_idx + 1}/{len(dataloader)} \
                    Loss D: {discriminator_loss:.4f}, Loss G: {generator_loss:.4f}')


    discriminator_loss_history.append(discriminator_batch_loss / (batch_idx + 1))
    generator_loss_history.append(generator_batch_loss / (batch_idx + 1))

    with torch.no_grad():
        
        fake_images = generator(test_noise)

        size_figure_grid = 10
        fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
        for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)

        for k in range(10*10):
            i = k // 10
            j = k % 10
            ax[i, j].cla()
            ax[i, j].imshow(im_convert(fake_images[k]))

        label = 'Epoch {0}'.format(epoch)
        fig.text(0.5, 0.04, label, ha='center')
        plt.savefig(result_folder + "/gan%03d.png" % epoch)
        plt.show(block=False)
        plt.pause(1.5)
        plt.close(fig)


# create gif, 2 frames per second
subprocess.call([
    'ffmpeg', '-framerate', '2', '-i', \
    result_folder + '/gan%03d.png', result_folder+'/output.gif'
])


# plot discriminator and generator loss history
# clear figure
plt.clf()
plt.plot(discriminator_loss_history, label='discriminator loss')
plt.plot(generator_loss_history, label='generator loss')
plt.legend()
plt.savefig(result_folder + "/loss-history.png")
plt.show()