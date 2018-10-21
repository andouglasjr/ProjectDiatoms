from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import os
import sys
import time
import copy

class ModelClass():
    
    def __init__(self, model_name="", num_classes = 50, feature_extract=False, use_pretrained=True, folder_names = None, device = None, log = None):
        self.model_name = model_name
        self.num_classes = num_classes
        self.feature_extract = feature_extract
        self.use_pretrained = use_pretrained
        if(device == None):
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        if (folder_names == None):
            self.folder_names = ['train_diatoms_3_class','val_diatoms_3_class']
        else:
            self.folder_names = folder_names
        
        self.model_ft = None
        self.log = log
        input_size = 0
        
        if model_name == "Resnet18":
            print("[!] Using Resnet18 model")
            self.model_ft = models.resnet18(pretrained=use_pretrained)
            self.set_parameter_requires_grad(self.model_ft, feature_extract)
            num_ftrs = self.model_ft.fc.in_features
            self.model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 244
               
        elif model_name == "Resnet50":
            print("[!] Using Resnet50 model")
            self.model_ft = models.resnet18(pretrained=use_pretrained)
            self.set_parameter_requires_grad(self.model_ft, feature_extract)
            num_ftrs = self.model_ft.fc.in_features
            self.model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 244
            
        else:
            print("[x] Invalid model name, exiting!")
            sys.exit()
                    
    def get_model(self):
        return self.model_ft
    
    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False
    
    def get_criterion(self):
        return nn.CrossEntropyLoss()      
    
    def get_optimization(self, model, lr, momentum):
        return optim.SGD(model.fc.parameters(), lr=lr, momentum=momentum)
    
    def get_scheduler(self, optimizer, step_size, gamma):
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    def save_model(self, model, name_model):
        torch.save(model, name_model)
        
    def load_model(self, name_model, localization):
        if(localization == 'cpu'):
            return torch.load(name_model, map_location=lambda storage, loc: storage)
        else:
            return torch.load(name_model)
        
    def get_device(self):
        return self.device
    
    def confusion_matrix(self, model, dataloaders, folder_name):
        was_training = model.training
        model.eval()
        correct = torch.zeros([1, 50], dtype=torch.int32, device = self.device)
        incorrect = torch.zeros([1, 50], dtype=torch.int32, device = self.device)
        results = torch.zeros([50, 50], dtype=torch.int32, device = self.device)
        cont_correct = 0
        cont_incorrect = 0

        with torch.no_grad():      
            for i, (inputs, labels) in enumerate(dataloaders[folder_name]):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for k in range(labels.size()[0]):
                    correct_class = labels[k]
                    if(preds[k] == labels[k]):
                        results[preds[k],preds[k]] +=1
                        correct[0,preds[k]] += 1
                        cont_correct += 1
                    else:
                        results[preds[k],labels[k]] +=1
                        incorrect[0,preds[k]] += 1
                        cont_incorrect += 1
                        
        return results, cont_correct, cont_incorrect
    
    def train_model(self, model, dataloaders, params, dataset_sizes, data):
        since = time.time()
        isKfoldMethod = False
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        model = model.to(self.get_device())
        criterion = self.get_criterion()
        #Get parameters of training
        lr = params['lr']
        momentum = params['momentum']
        num_epochs = params['num_epochs']
        step_size = params['step_size']
        gamma = params['gamma']
        set_criterion = params['set_criterion']
        
        #Setting parameters of training
        optimizer = self.get_optimization(model, lr, momentum)
        scheduler = self.get_scheduler(optimizer, step_size, gamma)
        
        if set_criterion:
            self.get_criterion()
        
        for epoch in range(num_epochs):
            self.log.log('Epoch {}/{}'.format(epoch, num_epochs - 1), 'l')
            self.log.log('-' * 10, 'l')

            # Each epoch has a training and validation phase
            for phase in self.folder_names:
                if phase ==  self.folder_names[0]:
                    scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.get_device())
                    labels = labels.to(self.get_device())
                    #print(inputs.shape)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase ==  self.folder_names[0]):
                        outputs = model(inputs)

                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase ==  self.folder_names[0]:
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)


                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                #if phase == 'train':
                    #loss = {'Acc':epoch_acc, 'Loss':epoch_loss}
                    #vis.plot_combine('Combine Plot',loss)
                content = '{} {:.4f} {:.4f}'.format(epoch, epoch_loss, epoch_acc)
                close = False
                if(epoch == num_epochs - 1):
                    close = True
                data.save_data_training(phase, content, close)
                self.log.log('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc), 'v')

                # deep copy the model
                if phase ==  self.folder_names[1] and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        self.log.log('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60), 'v')
        self.log.log('Best val Acc: {:4f}'.format(best_acc), 'v')

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model