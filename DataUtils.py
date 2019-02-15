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
from albumentations import MotionBlur
from DiatomsDatasetAug import DiatomsDatasetAug

class DataUtils():
    
    _folder_names = ['train_diatoms', 'test_diatoms_3_class']
    _folder_split = ['train', 'val']
    _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _mean, _std = [0.5018, 0.5018, 0.5018],[0.0837, 0.0837, 0.0837]
    _blur = MotionBlur(p=0.2)
    _data_transforms = transforms.Compose([transforms.CenterCrop(224),
                                           #_blur(),
                                          #transforms.Grayscale(1),
                                          transforms.ToTensor(),
                                          transforms.Normalize(_mean, _std)
                                          ])

    train_size, valid_size = 0, 0
    
    
    def __init__(self, folder_names = None, transformations=None, device=None, args = None):
        super(DataUtils, self).__init__()
        if args is None:
            print("Closing! Need some arguments!")
            self.data_dir = '../data/Dataset_5/Diatom50NEW_generated'
            self.batch_size = 256
            self.number_by_class = 21000
            self.older_model = False
            #exit()
        else:
            self.data_dir = args.data_dir
            self.batch_size = args.batch_size
            self.number_by_class = int(args.images_per_class)
            self.older_model = args.older_model
        
        self.device = self._device
        if device is not None: self.device = device
        self.transformations = self._data_transforms
        if transformations is not None: self.transformations = transformations
        self.folder_names = self._folder_names
        if folder_names is not None: self.folder_names = folder_names 
        self.aug = args.new_aug
        if self.older_model:
            print("Older model activated!")
            self.images_dataset = ImageFolderDiatoms(os.path.join(self.data_dir, self.folder_names[0]), self.transformations, number_by_class = self.number_by_class)
        else:
            self.images_dataset = DiatomsDatasetAug(os.path.join(self.data_dir, self.folder_names[0]), aug=self.aug, args = args)
              
    def get_image_datasets(self):
        return self.images_dataset
    
    def get_dataset_sizes(self):
        return self.train_size, self.valid_size
    
    def load_data(self, validation_split = .2, dataset_name = 'train'):
        if(dataset_name == 'train'):
            dataset = self.images_dataset
            batch_size = self.batch_size
            shuffle_dataset = True
            random_seed= 42
            size_dataset_used = 1 #Number of dataset training images that will be used (in percentage) 
            
            # Creating data indices for training and validation splits:
            dataset_size = len(dataset)
            print(dataset_size)
            indices = list(range(dataset_size))
            
            np.random.seed(random_seed)
            np.random.shuffle(indices)

            split = int(np.floor(validation_split * dataset_size))
            train_indices, val_indices = indices[split:], indices[:split]

            # Creating PT data samplers and loaders:
            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(val_indices)
            
            self.train_size = len(train_sampler)
            self.valid_size = len(valid_sampler)

            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                                       sampler=train_sampler, num_workers=2)
            validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                            sampler=valid_sampler, num_workers=2)
            dataloaders = {'train' : train_loader, 'val' : validation_loader}
            
        elif(dataset_name == 'test'):
            test_loader = torch.utils.data.DataLoader(self.images_dataset, batch_size=self.batch_size)
            dataloaders = {'test' : test_loader}

            
        return dataloaders
    
    def open_file_data(self, folder, net_name, lr, args):        
            localtime = args.time_training
            folder_save_results = args.save_dir+'/'+net_name+'/lr_'+str(lr)+'_'+str(args.time_training)+'/results'
            if not os.path.exists(folder_save_results):
                os.makedirs(folder_save_results)
            
            file_train = open(folder_save_results+'/data_train.dat','w')
            file_val = open(folder_save_results+'/data_val.dat','w')
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
                    
    def compute_mean(self, dataloaders, name_dir):
        pop_mean = []
        pop_std0 = []
        pop_std1 = []
        device = self.device
        toTensor = transforms.ToTensor()
        #image_datasets = datasets.ImageFolder(os.path.join(data_dir, name_dir))

        #dataloaders =  {name_dir: torch.utils.data.DataLoader(image_datasets[name_dir], 
        #                                           batch_size = self.batch_size, 
        #                                           shuffle = self.shuffle, 
        #                                           num_workers = self.num_workers)}
                    
        for i, sample in enumerate(dataloaders['train']):
            inputs, labels, filename = sample
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
    
    folder_name = ['train_diatoms']
    data_test = DataUtils(folder_names = folder_name)
    dataloaders_test = data_test.load_data()
    dataset_size_test = len(data_test.images_dataset)
    print(dataset_size_test)
    print(data_test.compute_mean(dataloaders_test, folder_name[0]))