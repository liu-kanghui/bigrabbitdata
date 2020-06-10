'''
Author: Kanghui Liu
Date: 4/11/2020
Reference: 
1. https://github.com/MorvanZhou/PyTorch-Tutorial/
2. https://github.com/rslim087a/PyTorch-for-Deep-Learning-and-Computer-Vision-Course-All-Codes-/
Standing on the shoulders of giants:)

Blog Tutorial: https://www.bigrabbitdata.com/pytorch-7-deep-neural-networks/
Google Colab： https://colab.research.google.com/drive/1tA-bgfnfyDPeSsHa_jDa-Ihywtw25OPq 
'''


import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import torch.nn as nn


data_points = 300
X, y = datasets.make_circles(n_samples=data_points, factor=0.3, random_state=5, noise=0.1)


class Model(nn.Module):
    def __init__(self, input_size, hidden_one, output_size):
      super().__init__() 
      self.linear = nn.Linear(input_size, hidden_one)
      self.hidden_layer_1 = nn.Linear(hidden_one, output_size)
    def forward(self, x):
      x = torch.sigmoid(self.linear(x))
      x = torch.sigmoid(self.hidden_layer_1(x))
      return x
    def predict(self, x):
      pred = self.forward(x)
      return np.where(pred<0.5, 0, 1)


def plot_decision_boundary(X, y):
  # Plot the decision boundary
  # Determine grid range in x and y directions and plus some padding 
  # x_span (50, 1)
  x_span = np.linspace(min(X[:, 0]), max(X[:, 0]))
  y_span = np.linspace(min(X[:, 1]), max(X[:, 1]))

  # XX (50, 50)
  XX, YY = np.meshgrid(x_span, y_span)

  # data (2500, 2)
  data = np.c_[XX.ravel(), YY.ravel()]

  # Pass data to predict method
  data_tensor = torch.FloatTensor(data)

  #output (2500, 1)
  output = model.predict(data_tensor)
  Z = output.reshape(XX.shape)

  plt.contourf(XX, YY, Z, cmap='spring')
  
  plt.scatter(X[y == 0, 0],
            X[y == 0, 1],
            c='blue',
            marker='v',
            label='cluster 0')

  plt.scatter(X[y == 1,0],
              X[y == 1,1],
              c='green',
              marker='o',
              label='cluster 1')
  plt.legend(loc='upper left')


torch.manual_seed(5)
model = Model(2, 4, 1)


criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.05)

X_tensor = torch.Tensor(X)
y_tensor = torch.Tensor(y).reshape(data_points, 1)
epochs = 200

for e in range(epochs):
  y_pred = model.forward(X_tensor)
  loss = criterion(y_pred, y_tensor)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  plt.cla()
  plot_decision_boundary(X, y)
  plt.text(0.5, 0, 'epoch: %d, Loss=%.4f' %(e+1, loss.data.numpy()), 
            fontdict={'size': 10, 'color':  'black'})
  plt.pause(0.00001)


# turn the interactive mode off
plt.ioff()
plt.show()

# prediction 
point_1 = torch.Tensor([0.025, 0.025])
point_2 = torch.Tensor([0.75, -0.75])

print ("Point A Red in the group:",  model.predict(point_1))
print ("Point B Yellow in the group:",  model.predict(point_2))

plt.plot(point_1.numpy()[0], point_1.numpy()[1], 'ro')
plt.plot(point_2.numpy()[0], point_2.numpy()[1], 'yv')
plot_decision_boundary(X, y)
plt.show()