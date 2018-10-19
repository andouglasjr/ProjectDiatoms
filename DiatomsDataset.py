from torchvision import datasets, models, transforms
import os

class DiatomsDataset():
    
    def __init__(self, file, data_dir, transform=None):
        self.diatoms_dataset = datasets.ImageFolder(os.path.join(data_dir, file))
        self.file = file
        self.data_dir = data_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.diatoms_dataset)
    
    def __getitem__(self, idx):
        diatoms_data = self.diatoms_dataset[idx]
        image = diatoms_data[0]
        sample = {'image': image, 'diatoms': diatoms_data[1]}
        
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        
        return sample
            
                
            
                                                            
            
            
    