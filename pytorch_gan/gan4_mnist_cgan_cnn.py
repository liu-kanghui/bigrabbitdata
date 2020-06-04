'''
Author: Kanghui Liu
Date: 5/30/2020

Blog in depth tutorial: https://www.bigrabbitdata.com/
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
    def __init__(self, image_channels, features, mode, label_features):
        super().__init__()

        # create 64 * 64 * 1 * 

        self.embed_layer = nn.Embedding(mode, 32 * 32)
        
        # 64 is intput image size 
        self.conv_0 = nn.Sequential( 
            # Batch_size x image_channels x 64 x 64
            nn.Conv2d(image_channels, features, 
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(features),
    
        )

        self.conv_1 = nn.Sequential(
            # Batch_size x features  x 32 x 32
            nn.Conv2d(features +1, features * 2, 
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(features*2)
        )

        self.conv_2 = nn.Sequential(
            # Batch_size x (features * 2) x 16 x 16
            nn.Conv2d(features * 2, features * 4, 
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(features *4)
        )

        self.conv_3 = nn.Sequential(
            # Batch_size x (features * 4) x 8 x 8
            nn.Conv2d(features * 4, features * 8, 
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(features*8)
        )

        self.out = nn.Sequential(
             # Batch_size x (features * 8) x 4 x 4
            nn.Conv2d(features * 8, image_channels, 
                      kernel_size=4, stride=1, padding=0),
            # Batch x 1 x 1 x 1
            nn.Sigmoid()
        )


    def forward(self, x, label):
        label = self.embed_layer(label).view(label.shape[0],
                                             1, 32, 32)
        x = self.conv_0(x)
        # concate on the color channel
        x = torch.cat([x, label], dim=1)

        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.out(x)
        return x


class Generator(nn.Module):
    '''
        Generator model
    '''
    def __init__(self, channels_noise, image_channels, features, mode, label_features):
        super().__init__()

        self.embed_layer = nn.Embedding(mode, 10)

        self.deconv_0 = nn.Sequential(
            # Batch_size x channels_noise x 1 x 1
            nn.ConvTranspose2d(channels_noise + 10, features * 8, 
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
            nn.BatchNorm2d(features * 2),
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

    def forward(self, x, label):
        label = self.embed_layer(label).view(label.shape[0],
                                             10 , 1, 1)
        # concate on the color channel
        x = torch.cat([x, label], dim=1)
        x = self.deconv_0(x)
        x = self.deconv_1(x)
        x = self.deconv_2(x)
        x = self.deconv_3(x)
        x = self.out(x)
        return x


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
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,)),
    ])


mode = 10
label_features = 10
dataset = datasets.FashionMNIST(root='dataset/', train=True, 
                        transform=transforms, download=True)
dataloader = DataLoader(dataset, batch_size=128, 
                        drop_last=True,
                        shuffle=True)



# Loading models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create discriminator and generator
discriminator = Discriminator(image_channels=1,
                              features = 10,
                              mode=mode,
                              label_features=label_features).to(device)
generator = Generator(noise_features, 
                      image_channels=1, 
                      features=10,
                      mode=mode,
                      label_features=label_features).to(device)

# Create 100 test_noise for visualizing how well our model perform.
test_noise = noise(100).to(device)
test_label = torch.arange(mode).repeat_interleave(10).to(device)

# Optimizers
lr = 0.0002
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()


# True and False Labels.  128 is the batch size
true_label = torch.ones(128).to(device)
false_label = torch.zeros(128).to(device)


# Create folder to hold result
result_folder = 'gan4-result'
if not os.path.exists(result_folder ):
    os.makedirs(result_folder )

# Training in action
print("Starting Training...")

num_epochs = 50
discriminator_loss_history = []
generator_loss_history = []

for epoch in range(1, num_epochs+1):
    discriminator_batch_loss = 0.0
    generator_batch_loss = 0.0

    for batch_idx, (data, real_label) in enumerate(dataloader):
        data = data.to(device)
        batch_size = data.shape[0]
        real_label = real_label.to(device)

        discriminator.zero_grad()

        # Train discriminator to get better at differentiate real/fake data
        # 1.1 Train discriminator on real data
        d_real_predict = discriminator(data, real_label)
        d_real_loss = criterion(d_real_predict.reshape(-1), true_label)
  

        # 1.2 Train discriminator on fake data from generator
        d_fake_noise = noise(batch_size).to(device)
        # Generate outputs and detach to avoid training the Generator on these labels
        d_fake_input = generator(d_fake_noise, real_label).detach()
        d_fake_predict = discriminator(d_fake_input, real_label)
        d_fake_loss = criterion(d_fake_predict.reshape(-1), false_label)

        # 1.3 combine real loss and fake loss for discriminator
        discriminator_loss = d_real_loss + d_fake_loss
        discriminator_batch_loss += discriminator_loss.item()
        discriminator_loss.backward()
        optimizerD.step()


        # Train generator to get better at deceiving discriminator
        g_fake_noise = noise(batch_size).to(device)
        g_fake_input = generator(g_fake_noise, real_label)
        generator.zero_grad()
        # Get prediction from discriminator
        g_fake_predict = discriminator(g_fake_input, real_label)
        generator_loss = criterion(g_fake_predict.reshape(-1), true_label)
        generator_batch_loss += generator_loss.item()
        generator_loss.backward()
        optimizerG.step()

        if (batch_idx + 1) % 100 == 0:

            print(f'Epoch [{epoch}/{num_epochs}]  Batch {batch_idx + 1}/{len(dataloader)} \
                    Loss D: {discriminator_loss:.4f}, Loss G: {generator_loss:.4f}')


    discriminator_loss_history.append(discriminator_batch_loss / (batch_idx + 1))
    generator_loss_history.append(generator_batch_loss / (batch_idx + 1))

    with torch.no_grad():
        
        fake_images = generator(test_noise, test_label)

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
plt.savefig(result_folder + "/loss-history.png")
plt.legend()
plt.show()