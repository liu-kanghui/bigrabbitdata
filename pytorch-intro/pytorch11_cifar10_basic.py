'''
Author: Kanghui Liu
Date: 4/30/2020
Reference: 
1. https://github.com/rslim087a/PyTorch-for-Deep-Learning-and-Computer-Vision-Course-All-Codes-/
Standing on the shoulders of giants:)

Blog Tutorial: https://www.bigrabbitdata.com/pytorch-11-classify-cifar-10-dataset-with-lenet/
Google Colab： https://colab.research.google.com/drive/1iPgimx30ne96WJ8C-gTBpVERwXSfNE9o 
'''



import torch
import matplotlib.pyplot as plt 
import numpy as np
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms

import PIL.ImageOps
import requests
from PIL import Image


transform  = transforms.Compose([transforms.Resize((32,32)),
                                 transforms.ToTensor()
                                ])
training_dataset = datasets.CIFAR10(root='./cifar10', train=True, 
                                  download=True, transform= transform)
validation_dataset = datasets.CIFAR10(root='./cifar10', train=False, 
                                  download=True, transform= transform) 

print ("length of training_dataset: ", len(training_dataset))
print ("length of validation_dataset: ", len(validation_dataset))


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Plot some sample data from training set
fig=plt.figure(figsize=(15, 5))

for idx in range(1, 11):
  fig.add_subplot(2, 5, idx)
  plt.title(classes[training_dataset[idx][1]])
  plt.imshow(transforms.ToPILImage()(training_dataset[idx][0]))
plt.tight_layout()
plt.show()


class LeNet(nn.Module):
    def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(in_channels=3, out_channels= 6, kernel_size=5)
      self.conv2 = nn.Conv2d(6, 16, 5)
      self.fc1 = nn.Linear(5*5*16, 120)
      self.fc2 = nn.Linear(120, 84)
      self.output = nn.Linear(84, 10)
    def forward(self, x):
      x = F.relu(self.conv1(x))
      x = F.max_pool2d(x, 2, 2)
      x = F.relu(self.conv2(x))
      x = F.max_pool2d(x, 2, 2)
      x = x.view(-1, 5*5*16)
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.output(x)
      return x


# If you don't have cuda on your computer
# training will be default to cpu
device = 'cpu'
if torch.cuda.is_available():
    device = torch.device('cuda')


model = LeNet().to(device)

training_loader = torch.utils.data.DataLoader(dataset=training_dataset, 
                                              batch_size=100,
                                              shuffle=True)
   
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, 
                                              batch_size=100,
                                              shuffle=True)


# Cross Entropy Loss with Adam Optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

epochs = 15

train_corrects_history = []
val_corrects_history = []

for e in range(epochs):

    train_corrects = 0.0 
    train_batch_loss = 0.0
    train_epoch_loss = 0.0
    val_corrects = 0.0
    val_epoch_loss = 0.0
    
    # loop through 60000 samples 100 at a time
    for batch_idx, data in enumerate(training_loader, start=1):
        inputs = data[0].to(device)
        labels = data[1].to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Return the index of the highest possibility
        # which are the predicted labels
        _, preds = torch.max(outputs, 1)
        train_batch_loss += loss.item()

        # sum up all the correct prediction
        # since (preds==labels).sum() is a tensor
        # we use item() to extract the number
        train_corrects += (preds == labels).sum().item()

        # print training loss every 100 mini-batch
        # train_batch_loss is the average loss for 100 mini-batch
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                   e + 1 , 
                   batch_idx * len(data[0]), 
                   len(training_loader.dataset),
                   100.* batch_idx * len(data[0]) / len(training_loader.dataset), 
                   train_batch_loss / 100 ))
            # accumulate loss for the epoch
            train_epoch_loss += train_batch_loss
            # reset the loss for every mini-batch
            train_batch_loss = 0.0
    else:
        # torch.no_grad deactivate the auograd engine, 
        # reduce memory usage and speed up computations
        with torch.no_grad():
            for val_data in validation_loader:
                val_inputs = val_data[0].to(device)
                val_labels = val_data[1].to(device)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)

                _, val_preds = torch.max(val_outputs, 1)
                val_epoch_loss += val_loss.item()
                val_corrects += (val_preds == val_labels).sum().item()


        # print result for every epoch 
        train_accuracy = 100. * train_corrects / len(training_loader.dataset)
        train_corrects_history.append(train_accuracy)
        # here batch_idx is the total number of mini-batch = 600
        train_epoch_loss /= batch_idx

        print('epoch :', (e+1))
        print('Train set: Accuracy: {}/{} ({:.0f}%), Average Loss: {:.6f}'.format(
                train_corrects, len(training_loader.dataset),
                train_accuracy, train_epoch_loss))


        
        val_accuracy = 100. * val_corrects / len(validation_loader.dataset)
        val_corrects_history.append(val_accuracy)
        val_epoch_loss /= batch_idx

        print('Validation set: Accuracy: {}/{} ({:.0f}%), Average Loss: {:.6f}'.format(
                val_corrects, len(validation_loader.dataset),
                val_accuracy, val_epoch_loss))



plt.plot(train_corrects_history, label='training accuracy')
plt.plot(val_corrects_history, label='validation accuracy')
plt.legend()
plt.show()


dataiter = iter(validation_loader)
images, labels = dataiter.next()
images = images.to(device)
labels = labels.to(device)
output = model(images)
_, preds = torch.max(output, 1)

fig = plt.figure(figsize=(10, 12))

for idx in np.arange(16):
  ax = fig.add_subplot(4, 4, idx+1, xticks=[], yticks=[])
  plt.imshow(transforms.ToPILImage()(images[idx].cpu()))
  ax.set_title("{} ({})".format(str(classes[preds[idx].item()]), 
                                str(classes[labels[idx].item()])), 
                                color=("green" if preds[idx]==labels[idx] else "red")
                                )
plt.show()


# Predict pictures from onlin
url = 'https://specials-images.forbesimg.com/imageserve/5d35eacaf1176b0008974b54/960x0.jpg?cropX1=790&cropX2=5350&cropY1=784&cropY2=3349'
response = requests.get(url, stream = True)
img = Image.open(response.raw)
plt.imshow(img)
plt.show()

img = transform(img)
img = img.unsqueeze(0).to(device)
output = model(img)
_, pred = torch.max(output, 1)
print("our predicted result is ", classes[pred.item()])



# Actual image our model see
url = 'https://specials-images.forbesimg.com/imageserve/5d35eacaf1176b0008974b54/960x0.jpg?cropX1=790&cropX2=5350&cropY1=784&cropY2=3349'
response = requests.get(url, stream = True)
img = Image.open(response.raw)

img = transform(img)
plt.imshow(transforms.ToPILImage()(img))
plt.show()