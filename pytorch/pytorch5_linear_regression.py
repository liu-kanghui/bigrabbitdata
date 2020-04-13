'''
Author: Kanghui Liu
Date: 4/6/2020
Reference: 
1. https://github.com/MorvanZhou/PyTorch-Tutorial/
2. https://github.com/rslim087a/PyTorch-for-Deep-Learning-and-Computer-Vision-Course-All-Codes-/
Standing on the shoulders of giants:)

Blog Tutorial: https://bigrabbitdata.com/pytorch-5-linear-regression/
Google Colab： https://colab.research.google.com/drive/1gqg2WY7FcpvIdJ5I4GL2J4L1VqcNTbIK
'''


import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# Neural Net
class LinearRegression(nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    self.linear = nn.Linear(in_features, out_features)

  def forward(self, x):
    prediction = self.linear(x)
    return prediction


# generate random data
x = torch.randn(100, 1) * 10
y = x + 5 * torch.randn(100, 1)

# use the same seed will produced the same initial random line 
torch.manual_seed(0)

model = LinearRegression(1, 1)

# Define the lost function
criterion = nn.MSELoss()
# if you set the learning rate too high
# it might never find the right direction.
# try to set the lr to 0.01 or 0.05 or 0.1 to see what will happen
optimizer = torch.optim.SGD(model.parameters(), lr = 0.0005)

epochs = 50

# turn the interactive mode on
plt.ion()

for e in range(epochs):
  optimizer.zero_grad()
  y_pred = model.forward(x)
  loss = criterion(y_pred, y)

  loss.backward()
  optimizer.step()

  # show the learning process
  plt.cla()
  plt.scatter(x, y)
  plt.plot(x, y_pred.data.numpy(), 'orange', lw=3)
  plt.text(0.5, 0, 'epoch: %d, Loss=%.4f' %(e+1, loss.item()), 
            fontdict={'size': 10, 'color':  'red'})
  plt.pause(0.2)


# turn the interactive mode off
plt.ioff()
plt.show()


