from torch.utils import data as data
from torchvision import transforms
import os
import glob
import os.path as osp
from PIL import Image
import torch
from matplotlib import pyplot as plt

from albumentations import (ToFloat, 
    CLAHE, RandomRotate90, Transpose, ShiftScaleRotate, Blur, OpticalDistortion, 
    GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, 
    MedianBlur, IAAPiecewiseAffine, IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, 
    Flip, OneOf, Compose
)
import numpy as np

class DiatomsDatasetAug(data.Dataset):
    
    def __init__(self, root, aug=False, args = None):
        self.filenames = []
        self.classes = []
        self.root = root
        self.aug = aug
        self.number_by_class = int(args.images_per_class)
        self.number_classes = int(args.classes_training)        
        
        #The mean,std and clases change due the number of classes that will be used in the training
        if(self.number_classes == 3): 
            classes = [27,41,42]
            mean, std = [0.5018, 0.5018, 0.5018],[0.0837, 0.0837, 0.0837]
        else:
            classes = [x for x in range(1,51)]
            mean, std = [0.5017, 0.5017, 0.5017],[0.1057, 0.1057, 0.1057]
        
        print(classes)
        if self.aug:
            self.transform_compose_new = MotionBlur(blur_limit=100, p=1)
        
        self.transform_compose = transforms.Compose([transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean, std)
                                          ])
        
        cont = 1
        for num_class in classes:
            filenames = glob.glob(osp.join(root+'/'+str(num_class), '*.png'))
            for fn in filenames:
                self.filenames.append(fn)
                if(self.number_classes==3):
                    self.classes.append(classes.index(num_class)) 
                else:
                    self.classes.append(num_class-1) 
                if(cont == self.number_by_class):
                    cont = 1
                    break
                cont=cont+1
        
        self.len = len(self.filenames)
    
    def __getitem__(self, index):
        image = Image.open(self.filenames[index])
        classe = self.classes[index]
        image = self.transform_compose(image)
        if self.aug:
            data = {"image": np.array(image)}
            image = self.transform_compose_new(**data)['image']    
        
        return image, classe, self.filenames[index]
    
    def __len__(self):
        return self.len
                