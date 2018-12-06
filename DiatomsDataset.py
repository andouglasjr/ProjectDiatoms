from torchvision import datasets, models, transforms
import os
from ImageFolderDiatoms import ImageFolderDiatoms

class DiatomsDataset():
    
    def __init__(self, file, data_dir, transform=None):
        self.diatoms_dataset = ImageFolderDiatoms(os.path.join(data_dir, file))
        self.file = file
        self.data_dir = data_dir
        self.transform = transform
        self.vector_transform = [1 ,10,11,12,13,
                                 14,15,16,17,18,
                                 19,2 ,20,21,22,
                                 23,24,25,26,27,
                                 28,29,3 ,30,31,
                                 32,33,34,35,36,
                                 37,38,39,4 ,40,
                                 41,42,43,44,45,
                                 46,47,48,49,5 ,
                                 50, 6, 7, 8, 9]
    
    def __len__(self):
        return len(self.diatoms_dataset)
    
    def __getitem__(self, idx):
        diatoms_data_image, diatoms_data_name = self.diatoms_dataset[idx]
        image_transformed = diatoms_data_image[0]
        if self.transform:
             image_transformed = self.transform[self.file](image_transformed)
            
        sample = {'image': image_transformed, 'diatoms': self.vector_transform[diatoms_data_image[1]], 'file_name' : diatoms_data_name[0]}
        
        
        
        return sample
            
                
            
                                                            
            
            
    