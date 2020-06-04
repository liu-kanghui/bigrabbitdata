'''
Author: Kanghui Liu
Date: 5/28/2020

Blog in depth tutorial: https://www.bigrabbitdata.com/
'''

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import itertools

import matplotlib.pyplot as plt 
import numpy as np


class Discriminator(nn.Module):
    '''
        Discriminator model
        mode: different classes from input
        label_features: num features for label
    '''
    def __init__(self, in_features, out_features, mode, label_features):
        super().__init__()

        self.embed_layer = nn.Embedding(mode, label_features)

        self.hidden_0 = nn.Sequential( 
            nn.Linear(in_features + label_features, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden_1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden_2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(256, out_features),
            nn.Sigmoid()
        )

    def forward(self, x, label):
        label = self.embed_layer(label)
        x = torch.cat([x, label], dim=1)
        x = self.hidden_0(x)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.out(x)
        return x


class Generator(nn.Module):
    '''
        Generator model
    '''
    def __init__(self, in_features, out_features, mode, label_features):
        super().__init__()

        self.embed_layer = nn.Embedding(mode, label_features)

        self.hidden_0 = nn.Sequential(
            nn.Linear(in_features + label_features,  256),
            nn.ReLU(),
            nn.Dropout(0.3)
            
        )
        self.hidden_1 = nn.Sequential(            
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.hidden_2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            nn.Linear(1024, out_features),
            nn.Tanh()
        )

    def forward(self, x, label):
        label = self.embed_layer(label)
        x = torch.cat([x, label], dim=1)
        x = self.hidden_0(x)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.out(x)
        return x


def noise(data_size):
    '''
    Generates data_size number of random noise
    '''
    noise_features = 100
    # create a normal distribution of noise
    n = torch.randn(data_size, noise_features)
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
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,)),
    ])


batch_size = 128
mode = 10
label_features = 10
dataset = datasets.MNIST(root='dataset/', train=True, 
                        transform=transforms, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, 
                        drop_last=True,
                        shuffle=True)



# Loading models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create discriminator and generator
discriminator = Discriminator(in_features=784, 
                              out_features=1,
                              mode=mode,
                              label_features=label_features).to(device)
generator = Generator(in_features=100,
                      out_features=784,
                      mode=mode,
                      label_features=label_features).to(device)


# Create 100 test_noise , 10 x  for each mode(label)
test_noise = noise(100).to(device)
test_label = torch.arange(mode).repeat_interleave(10).to(device)


# Optimizers
lr = 0.0002
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

true_label = torch.ones(batch_size, 1).to(device)
false_label = torch.zeros(batch_size, 1).to(device)


# Training in action
print("Starting Training...")

plt.ion()
plt.show()
num_epochs = 100
for epoch in range(1, num_epochs+1):
    for batch_idx, (data, real_label) in enumerate(dataloader):
        data = data.to(device)
        batch_size = data.shape[0]
        real_label = real_label.to(device)
  
        discriminator.zero_grad()

        # Train discriminator to get better at differentiate real/fake data
       
        # 1.1 Train discriminator on real data
        d_real_predict = discriminator(data.view(data.shape[0], -1), real_label)
        d_real_loss = criterion(d_real_predict, true_label)
  

        # 1.2 Train discriminator on fake data from generator
        d_fake_noise = noise(batch_size).to(device)
        # Generate outputs and detach to avoid backward() on the Generator
        d_fake_input = generator(d_fake_noise, real_label).detach()
        d_fake_predict = discriminator(d_fake_input, real_label)
        d_fake_loss = criterion(d_fake_predict, false_label)

        # 1.3 combine real loss and fake loss for discriminator
        discriminator_loss = d_real_loss + d_fake_loss
        discriminator_loss.backward()
        optimizerD.step()

        # Train generator to get better at deceiving discriminator
        g_fake_noise = noise(batch_size).to(device)
        g_fake_input = generator(g_fake_noise, real_label)
        generator.zero_grad()
        # Get prediction from discriminator
        g_fake_predict = discriminator(g_fake_input, real_label)
        generator_loss = criterion(g_fake_predict, true_label)
        generator_loss.backward()
        optimizerG.step()

        if (batch_idx + 1) % 200 == 0:

            print(f'Epoch [{epoch}/{num_epochs}] Batch {batch_idx + 1}/{len(dataloader)} \
                    Loss D: {discriminator_loss:.4f}, loss G: {generator_loss:.4f}')


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
            ax[i, j].imshow(im_convert(fake_images[k].view(1,28,28)))

        label = 'Epoch {0}'.format(epoch)
        fig.text(0.5, 0.04, label, ha='center')
        plt.pause(0.001)
        # plt.savefig("file%02d.png" % epoch)


    
