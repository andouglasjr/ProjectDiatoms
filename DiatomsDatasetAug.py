from torch.utils import data as data
from torchvision import transforms
import os
import glob
import os.path as osp
from PIL import Image
from albumentations import (ToFloat, 
    CLAHE, RandomRotate90, Transpose, ShiftScaleRotate, Blur, OpticalDistortion, 
    GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, 
    MedianBlur, IAAPiecewiseAffine, IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, 
    Flip, OneOf, Compose
)
import numpy as np

class DiatomsDatasetAug(data.Dataset):
    
    def __init__(self, root, aug=False):
        self.filenames = []
        self.root = root
        self.aug = aug
        
        if self.aug:
            self.transform = OneOf([
                                CLAHE(clip_limit=2),
                                IAASharpen(),
                                RandomRotate90(),
                                IAAEmboss(),
                                Transpose(),
                                RandomContrast(),
                                RandomBrightness(),
                            ], p=0.3)
        else:
            self.transform = transforms.ToTensor()
            
        for num_class in range(1, 51):
            filenames = glob.glob(osp.join(root+'/'+str(num_class), '*.png'))
            for fn in filenames:
                self.filenames.append(fn)
        
        self.len = len(self.filenames)
    
    def __getitem__(self, index):
        image = Image.open(self.filenames[index])
        if self.aug:
            data = {"image": np.array(image)}
            image = self.transform(**data)['image']
            images = np.transpose(image)
            return images
        else:
            return self.transform(image)
    
    def __len__(self):
        return self.len
                