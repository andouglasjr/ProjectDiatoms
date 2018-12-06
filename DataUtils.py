import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
from matplotlib import pyplot as plt
from ImageFolderDiatoms import ImageFolderDiatoms
from DiatomsDataset import DiatomsDataset
import csv
import math
from torch.utils.data.sampler import SubsetRandomSampler

class DataUtils():

    def __init__(self, list_of_name_folders, data_dir, transformations=None, batch_size = 128, shuffle = True, num_workers = 0, net_name='', device=None, phase='train'):
        super(DataUtils, self).__init__()
        self.list_of_name_folders = list_of_name_folders
        self.data_dir = data_dir
        self.folder_split = ['train', 'val']
        if(phase == 'train'):
            self.images_dataset = ImageFolderDiatoms(os.path.join(data_dir, self.list_of_name_folders[0]), transformations[self.list_of_name_folders[0]], number_by_class = 21000)
        else:
            self.images_dataset = ImageFolderDiatoms(os.path.join(data_dir, self.list_of_name_folders[1]), transformations[self.list_of_name_folders[1]])
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.device = device
        self.net_name = net_name       
        
        
    def get_image_datasets(self):
        return self.images_dataset
    
    def get_one_image_dataset(self, name_dir):
        return self.images_dataset[name_dir]
    
    def get_dataset_size(self):
        return len(self.images_dataset) 
    
    def load_data(self, validation_split = .2, dataset_name = 'train'):
        if(dataset_name == 'train'):
            dataset = self.images_dataset
            batch_size = self.batch_size
            shuffle_dataset = True
            random_seed= 42
            size_dataset_used = 0.01 #Number of dataset training images that will be used (in percentage) 
            
            # Creating data indices for training and validation splits:
            dataset_size = len(dataset)
            print(dataset_size)
            indices = list(range(dataset_size))
            
            np.random.seed(random_seed)
            np.random.shuffle(indices)
            
            #if size_dataset_used > 0:    
            #    new_split = int(np.floor(dataset_size * size_dataset_used))
            #    indices = indices[:new_split]
            #    dataset_size = len(indices)
                
            split = int(np.floor(validation_split * dataset_size))
            train_indices, val_indices = indices[split:], indices[:split]

            # Creating PT data samplers and loaders:
            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(val_indices)

            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                                       sampler=train_sampler, num_workers=2)
            validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                            sampler=valid_sampler, num_workers=2)
            dataloaders = {'train' : train_loader, 'val' : validation_loader}
            print("Already loaded all pictures!")
        elif(dataset_name == 'test'):
            test_loader = torch.utils.data.DataLoader(self.images_dataset, batch_size=self.batch_size)
            dataloaders = {'test' : test_loader}

            
        return dataloaders
    
    def open_file_data(self, folder, net_name, lr, drop_rate):
            file_train = open(folder+'/'+net_name+'/lr_'+str(lr)+'/data_train.dat','w')
            file_val = open(folder+'/'+net_name+'/lr_'+str(lr)+'/data_val.dat','w')
            self.results_files = {'train' : file_train, 'val' : file_val}
            
    
    def save_data_training(self, phase, content, close = False):
        self.results_files[phase].write('{}\n'.format(content))
        if close:
            self.results_files[phase].close()    
    
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
            
    def save_results(self, results, correct, incorrect, correct_class, log, incorrect_images, show_filename_image = False):
        log.log("Confusion Matrix Analyzes", 'l')
        results = results.to("cpu", torch.int32)
        log.log("Number Total of Test: {}".format(np.sum(results.numpy())), 'v')
        acc = (correct/(correct+incorrect))*100
        log.log("Accuracy: {}, Correct Number: {}, Incorrect Number: {}".format(acc, correct, incorrect), 'v')
        #   log.log("{}".format(results), 'v')
        res = torch.zeros([len(correct_class), len(correct_class)], dtype=torch.int32)
        cont_i = 0
        cont_j = 0
        #print(sorted(correct_class))
        for i in sorted(correct_class):
            cont_j = 0
            for j in sorted(correct_class):
                res[cont_i,cont_j] = results[i,j]
                cont_j += 1
            cont_i += 1                        
        
        if(show_filename_image):
            for i in range(1,len(incorrect_images)):
                log.log("Predict Class: {} -> Label Class: {} - Image Name: {}".format(incorrect_images[i]['class'], incorrect_images[i]['correct_class'], incorrect_images[i]['filename']), 'e')      
        
        #res = results[results != 0]
        log.log("{}".format(res), 'v')
        log.log("Diagonal Sum: {}".format(np.trace(results)), 'v')
        log.log("Classes confusions:", 'l')
        for i in range(50):
            for j in range(50):
                if(results[i,j] >= 1 and i != j):
                    log.log("Predict Class: {} -> Label Class: {} - Quantity {}".format(i, j, results[i,j].tolist()), 'e')      
                    
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
        return torch.from_numpy(pop_mean), torch.from_numpy(pop_std0)
    

if __name__ == "__main__":
    
    
    list_of_name_folders = ['train_diatoms_3_class_simulate_all','val_diatoms_3_class_simulate_all', 'test_diatoms_3_class']
    mean, std = [0.5018, 0.5018, 0.5018],[0.0837, 0.0837, 0.0837]
    
    data_transforms = {
        list_of_name_folders[0]: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(180),
            #transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            #transforms.Normalize([0.493], [0.085])
        ]),
        list_of_name_folders[1]: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            #transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            #transforms.Normalize([0.496], [0.07])
        ]),
         list_of_name_folders[2]: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            #transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            #transforms.Normalize([0.496], [0.07])
        ])
    }
    
    data_dir = '../data/Dataset_4'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    data = DataUtils(list_of_name_folders, data_dir, data_transforms, net_name = 'Resnet50', device = device)
    dataloaders = data.load_data(data.images_dataset)
    print(len(dataloaders['train']))
    
    