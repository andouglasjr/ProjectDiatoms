from urllib.request import urlopen
import numpy as np
import cv2
from matplotlib import pyplot as plt
from DataUtils import DataUtils
from torchvision import datasets, models, transforms
from DiatomsDatasetAug import DiatomsDatasetAug as diatoms
import torch
import torchvision
from torch.utils import data as D
from PIL import Image


from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose
)

def augment_and_show(aug, image):
    image = aug(image=image)['image']
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.show()
    

dir_path = '../data/train_'
ddimg = diatoms(dir_path, aug=True)
print(ddimg.len)


loader = D.DataLoader(ddimg, batch_size=50, shuffle=False, num_workers=0)

dataiter = iter(loader)
images, classes = dataiter.next()
print(images, classes)


plt.figure(figsize=(16,8))
batch_tensor = images.unsqueeze(1)
print(batch_tensor.shape)
grid_image = torchvision.utils.make_grid(batch_tensor, nrow=10)
print(grid_image.shape)

plt.imshow(grid_image.permute(1,2,0))  
plt.show()


#data = DataUtils(['train_'])
#data_dataset = data.get_image_datasets()

