'''
Author: Kanghui Liu
Reference: 
1. https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/301_regression.py
2. https://github.com/rslim087a/PyTorch-for-Deep-Learning-and-Computer-Vision-Course-All-Codes-/blob/master/PyTorch%20for%20Deep%20Learning%20and%20Computer%20Vision%20Course%20(All%20Codes)/Linear_Regression.ipynb
Standing on the shoulders of giants:)
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

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

epochs = 50
losses = []

# turn the interactive mode on
plt.ion()

for e in range(epochs):
  optimizer.zero_grad()
  y_pred = model.forward(x)
  loss = criterion(y_pred, y)

  # uncomment to print the lossf
  # print ("epoch:", e, "loss: ", loss.item())
  losses.append(loss)
  loss.backward()
  optimizer.step()

  # show the learning process
  plt.cla()  # clear axes.
  plt.scatter(x, y)
  plt.plot(x, y_pred.data.numpy(), 'orange', lw=3)
  plt.text(0.5, 0, 'epoch: %d, Loss=%.4f' %(e+1, loss.data.numpy()), 
            fontdict={'size': 10, 'color':  'black'})
  plt.pause(0.2)


# turn the interactive mode off
plt.ioff()
plt.show()




