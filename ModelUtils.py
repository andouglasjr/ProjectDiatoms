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
import utils
from sklearn.metrics import f1_score
from CenterLoss import CenterLoss
import csv
from matplotlib import pyplot as plt
from DiatomsNetwork import DiatomsNetwork
import torchvision
from FullyConnectedCapsuled import FullyConnectedCapsuled
import pandas as pd
from DataUtils import DataUtils
from sklearn.metrics import accuracy_score
import utils

class ModelUtils():
    
    global best_loss
    global cont_to_stop
    
    best_loss = 1000
    cont_to_stop = 0
    
    def set_parameter_requires_grad(model, feature_extracting, num_of_layers):  
        if(feature_extracting):
            child_counter = 0
            for child in model.children():
                if child_counter < num_of_layers:
                    print("child ",child_counter," was frozen")
                    for param in child.parameters():
                        param.requires_grad = False
                elif child_counter == num_of_layers:
                    children_of_child_counter = 0
                    for children_of_child in child.children():
                        if children_of_child_counter < 1:
                            for param in children_of_child.parameters():
                                param.requires_grad = False
                            print('child ', children_of_child_counter, 'of child',child_counter,' was frozen')
                        else:
                            print('child ', children_of_child_counter, 'of child',child_counter,' was not frozen')
                        children_of_child_counter += 1
                else:
                    print("child ",child_counter," was not frozen")
                child_counter += 1
    
    def get_criterion(loss_function):
        if loss_function == 'softmax':
            return nn.Softmax()
        elif loss_function == 'cross_entropy':
            return nn.CrossEntropyLoss()
        else:
            print('Please, what is the loss function?')
            exit(0)
    
    def get_optimization(model, lr, momentum):
            return optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    def get_scheduler(optimizer, step_size, gamma):
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    def save_model(model, name_model):
        torch.save(model, name_model)
        
    def load_model(name_model, localization):
        if(localization == 'cpu'):
            return torch.load(name_model, map_location=lambda storage, loc: storage)
        else:
            return torch.load(name_model)
        
    def get_device():
        return self.device
    
    def update_correct_class(l1,l2):
        if ((l1 == l2).all()):
            return l1
        if(len(l2)>len(l1)):
            l1_ = l2
            l2_ = l1
        l1_ = set(l1)
        l2_ = set(l2)
        new_list = l1 + list(l2_ - l1_)
        return sorted(new_list) 
    
    def earlier_stop(loss):
        global best_loss
        global cont_to_stop
        #print(self.best_loss, loss)
        if(best_loss <= loss):
            cont_to_stop += 1
        else:
            best_loss = loss
            cont_to_stop = 0   
            
        if(cont_to_stop == 2):
            return True
        return False
    
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
