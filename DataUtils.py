import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import matplotlib.pyplot as plt

class DataUtils():

    def __init__(self, list_of_name_folders, data_dir, transformations=None, batch_size = 64, shuffle = True, num_workers = 4, net_name='', device=None):
        super(DataUtils, self).__init__()
        self.list_of_name_folders = list_of_name_folders
        self.data_dir = data_dir
        if transformations == None:
            self.image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x)) for x in self.list_of_name_folders}
        else:
            self.image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), 
                                                           transformations[x]) for x in self.list_of_name_folders}
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        self.device = device
        
        #init results files
        self.net_name = net_name
        str_name_file = self.net_name 
        if not str_name_file:
            file_train = open('results/'+self.net_name+'_data_train.dat','w')
            file_val = open('results/'+self.net_name+'_data_val.dat','w')
            self.results_files = {self.list_of_name_folders[0] : file_train, self.list_of_name_folders[1] : file_val}
        
        
        
    def get_all_image_datasets(self):
        return self.image_datasets
    
    def get_one_image_dataset(self, name_dir):
        return self.image_datasets[name_dir]
    
    def get_dataset_size(self):
        return {x: len(self.image_datasets[x]) for x in self.list_of_name_folders}
    
    def load_data(self, image_datasets):
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = self.batch_size, 
                                                      shuffle = self.shuffle, num_workers = self.num_workers) 
               for x in self.list_of_name_folders}
        return dataloaders
    
    def save_data_training(self, phase, content, close = False):
        self.results_files[phase].write('{}\n'.format(content))
        if close:
            self.results_files[phase].close()
    
    def log(self, content, close = False):
        print(content)
        file_log.write(content)
        if close:
            file_log.close()
        
    
    def set_normalization(self, tensor, mean = [0.496, 0.496, 0.496], std = [0.07, 0.07, 0.07]):
        normalization = transforms.Normalize(mean, std)
        normalization(tensor)
        
    def set_rotation(self, tensor, angle):
        rotation = transforms.RandomRotation(angle)
        rotation(tensor)
        
    def imshow(self, inp=None, title=None, mean = [0.496, 0.496, 0.496], std = [0.07, 0.07, 0.07]):
        if(inp is not None):
            """Imshow for Tensor."""
            inp = inp.numpy().transpose((1, 2, 0))
            inp = std * inp + mean
            inp = np.clip(inp, 0, 1)
            plt.imshow(inp)
            if title is not None:
                plt.title(title)
            plt.pause(0.001)  # pause a bit so that plots are updated
        else:
            print("Inp is None!")
            
    def compute_mean(self, image_datasets, name_dir):
        pop_mean = []
        pop_std0 = []
        pop_std1 = []
        device = self.device
        toTensor = transforms.ToTensor()
        #image_datasets = datasets.ImageFolder(os.path.join(data_dir, name_dir))

        dataloaders =  {name_dir: torch.utils.data.DataLoader(image_datasets[name_dir], 
                                                   batch_size = self.batch_size, 
                                                   shuffle = self.shuffle, 
                                                   num_workers = self.num_workers)}
                       
        for i, (inputs, labels) in enumerate(dataloaders[name_dir]):
            # shape (batch_size, 3, height, width)
            inputs = inputs.to(device)
            numpy_image = inputs.cpu().numpy()

            # shape (3,)
            batch_mean = np.mean(numpy_image, axis=(0,2,3))
            batch_std0 = np.std(numpy_image, axis=(0,2,3))
            batch_std1 = np.std(numpy_image, axis=(0,2,3), ddof=1)

            pop_mean.append(batch_mean)
            pop_std0.append(batch_std0)
            pop_std1.append(batch_std1)

        # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
        pop_mean = np.array(pop_mean).mean(axis=0)
        pop_std0 = np.array(pop_std0).mean(axis=0)
        pop_std1 = np.array(pop_std1).mean(axis=0)

        #print("Mean: $f, Std0: $f, Std1: $f", (pop_mean,pop_std0,pop_std1))
        return pop_mean, pop_std0
        

    
    
        