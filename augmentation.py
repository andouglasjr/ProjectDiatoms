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
import os
import glob
import os.path as osp


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
    

dir_path = '../data/Dataset_5/Diatom50NEW_generated/test_diatoms_3_class'
ddimg = diatoms(dir_path, aug=False)
print(ddimg.len)


loader = D.DataLoader(ddimg, batch_size=120, shuffle=False, num_workers=0)

dataiter = iter(loader)
images, classes,filename = dataiter.next()
#images = images.repeat(1,3,1,1)
for f in filename:
    print(f)


plt.figure(figsize=(16,8))
#batch_tensor = images.unsqueeze(1)
print(images.shape)
grid_image = torchvision.utils.make_grid(images, nrow=10)
print(grid_image.shape)

plt.imshow(grid_image.permute(1,2,0))  
plt.show()


#data = DataUtils(['train_'])
#data_dataset = data.get_image_datasets()

