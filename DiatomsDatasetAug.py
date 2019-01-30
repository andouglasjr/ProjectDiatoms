from torch.utils import data as data
from torchvision import transforms
import os
import glob
import os.path as osp
from PIL import Image
import torch

from albumentations import (ToFloat, 
    CLAHE, RandomRotate90, Transpose, ShiftScaleRotate, Blur, OpticalDistortion, 
    GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, 
    MedianBlur, IAAPiecewiseAffine, IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, 
    Flip, OneOf, Compose
)
import numpy as np

class DiatomsDatasetAug(data.Dataset):
    
    def __init__(self, root, aug=False, number_by_class = 200):
        self.filenames = []
        self.classes = []
        self.root = root
        self.aug = aug
        self.number_by_class = number_by_class
        
        
        if self.aug:
            self.transform = Compose([ OneOf([
                                #CLAHE(clip_limit=2),
                                #IAASharpen(),
                                #RandomRotate90(),
                                #IAAEmboss(),
                                #Transpose(),
                                #RandomContrast(),
                                MotionBlur(p=1),
                                #Blur(),
                                #RandomBrightness(),
                            ], p=1)], p=1)
            self.transform_crop = transforms.CenterCrop(224)
        else:
            self.transform = transforms.ToTensor()
        
        cont = 1
        for num_class in range(1, 51):
            filenames = glob.glob(osp.join(root+'/'+str(num_class), '*.png'))
            for fn in filenames:
                self.filenames.append(fn)
                self.classes.append(num_class)
                if(cont == number_by_class):
                    cont = 1
                    break
                cont=cont+1
        
        self.len = len(self.filenames)
    
    def __getitem__(self, index):
        image = Image.open(self.filenames[index])
        classe = self.classes[index]
        if self.aug:
            image = self.transform_crop(image)
            data = {"image": np.array(image)}
            image = self.transform(**data)['image']
            
            #image = transforms.CenterCrop(224)(image)
            #images = np.transpose(image, (2,0,1))
            return image, classe, self.filenames[index]
        else:
            return self.transform(image), classe, self.filenames[index]
    
    def __len__(self):
        return self.len
                