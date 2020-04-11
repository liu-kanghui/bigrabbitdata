import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import torch.nn as nn


centers = [[1, 0], [5, 0]]
X, y = datasets.make_blobs(random_state=5, centers=centers)


class Model(nn.Module):
    def __init__(self, input_size, output_size):
      super().__init__() 
      self.linear = nn.Linear(input_size, output_size)
    def forward(self, x):
      pred = self.linear(x)
      pred = torch.sigmoid(pred)
      return pred

torch.manual_seed(5)
model = Model(2, 1)

[w, b] = model.parameters()
w1, w2 = w.view(2)

# extract the number of weight and bias.
def get_parameters():
  return w1.item(), w2.item(), b[0].item()


def plot_graph():
  w1, w2, b1 = get_parameters()
  # To plot a line, we just need two data points
  x1 = np.array([-3, 7]) 
  y1 = w1*x1 + b1
  plt.plot(x1, y1, color='red')
  plt.scatter(X[y == 0, 0],
            X[y == 0, 1],
            c='blue',
            marker='v',
            label='cluster 1')

  plt.scatter(X[y == 1,0],
              X[y == 1,1],
              c='orange',
              marker='o',
              label='cluster 2')
  plt.legend(loc='upper left')


criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.05)


X_tensor = torch.Tensor(X)
y_tensor = torch.Tensor(y).reshape(100, 1)
epochs = 5

for e in range(epochs):
  y_pred = model.forward(X_tensor)
  loss = criterion(y_pred, y_tensor)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  plt.cla()
  plot_graph()
  plt.text(0.5, 0, 'epoch: %d, Loss=%.4f' %(e+1, loss.data.numpy()), 
            fontdict={'size': 10, 'color':  'red'})

  plt.pause(0.00001)


# turn the interactive mode off
plt.ioff()
plt.show()
