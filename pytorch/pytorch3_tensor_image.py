'''
Author: Kanghui Liu
Date: 3/14/2020

Blog Tutorial: https://www.bigrabbitdata.com/pytorch-3-tensor-and-images/
Google Colab： https://colab.research.google.com/drive/1hWVeohqQhaSIQLEn23fYYSQnCOi6SQxK
'''


import random
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToTensor, ToPILImage



# create a 50 * 50 2d array filled with random int value from [0, 255]
flat_image = np.array([random.randint(0, 255) for i in range(2500)])
square_image = flat_image.reshape(50,50)


# Because plt is displaying a color image (3 dimensions) by default, 
# since our square image is 2 dimension, it will auto-fill it to 3 dimension
# by default. 
# So we need to specify that we want it to display a 'grey' image.
plt.imshow(square_image, cmap="gray")
plt.show()


print (type(square_image))


# Convert np.array to tensor
tensor_square_image = torch.Tensor(square_image)

print (type(tensor_square_image))


# convert tensor to PIL Image 
pil_image = ToPILImage()(tensor_square_image)
print (type(pil_image))

# convert PIL Image to tensor
back_to_tensor_image = ToTensor()(pil_image)
print (type(back_to_tensor_image))