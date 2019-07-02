import csv
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
from DataLogger import DataLogger

class TestingClass():
    
    def __init__(self, model, data, args):
        self.model = model
        self.dataloaders = data.load_data(dataset_name = 'test') 
        self.data = data
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.folder_name = "test"
        self.class_names = self.data.get_image_datasets().classes
        self.args = args
        self.log = DataLogger.getInstance(args)
        
        self.vector_transform = [1 ,10,11,12,13, 14,15,16,17,18, 19,2 ,20,21,22, 23,24,25,26,27, 28,29,3 ,30,31,
                         32,33,34,35,36, 37,38,39,4 ,40, 41,42,43,44,45, 46,47,48,49,5, 50, 6, 7, 8, 9]
        
        self.vector_transform_old = [27, 41, 42]
    
    def results(self, y_test, y_pred):
        self.log.log("F1 SOCRE:", 'l')
        self.log.log("Macro: {}".format(f1_score(y_test, y_pred, average='macro')), 'v')
        self.log.log("Micro: {}".format(f1_score(y_test, y_pred, average='micro')), 'v')
        self.log.log("Weighted: {}".format(f1_score(y_test, y_pred, average='weighted')), 'v')
        self.log.log("For all analyzed classes: {}".format(f1_score(y_test, y_pred, average=None)), 'v')
            
        class_names = sorted(np.array(list(set(np.array(y_pred)))))
        self.data.confusion_matrix_skt(y_test = y_test, y_pred = y_pred, class_names=class_names, noPrint = False)
        accuracy = accuracy_score(y_test, y_pred)
        self.log.log("Accuracy: {}".format(accuracy * 100.0), 'v')
        
    def test_ensemble(self, models):
        y_pred = []
        y_test = []
        correct_class = []
        
        with torch.no_grad():      
            
            for i, sample in enumerate(self.dataloaders[self.folder_name]):
                inputs, labels, filename, shape = sample
                inputs = inputs.repeat(1,3,1,1)
                
                correct_class = np.concatenate((correct_class, labels),0)
                correct_class = sorted(np.array(list(set(np.array(correct_class,dtype=np.int16))),dtype=np.int16))            
                
                x_1 = models[0](inputs)
                x_2 = models[1](inputs)
                x = torch.cat((x_1,x_2), dim=1)
                
                #Ensemble
                outputs = self.model.predict(x)
                print(outputs)
                preds = outputs
                
                #print(preds)
                preds = [(p.item()+1) for p in preds]
                y_pred = np.concatenate((y_pred,preds),0)
                y_test = np.concatenate((y_test,labels),0)
            y_test = [l + 1 for l in y_test]
                
            self.results(y_test, y_pred)
        
    def test(self):            
        y_pred = []
        y_test = []
        correct_class = []
        
        with torch.no_grad():      
            
            for i, sample in enumerate(self.dataloaders[self.folder_name]):
                inputs, labels, filename, shape = sample
                inputs = inputs.repeat(1,3,1,1)
                
                correct_class = np.concatenate((correct_class, labels),0)
                correct_class = sorted(np.array(list(set(np.array(correct_class,dtype=np.int16))),dtype=np.int16))            
                
                outputs = self.model(inputs)
                
                _, preds = torch.max(outputs, 1)
                
                #print(preds)
                preds = [(p.item()+1) for p in preds]
                y_pred = np.concatenate((y_pred,preds),0)
                y_test = np.concatenate((y_test,labels),0)
            y_test = [l + 1 for l in y_test]
                
            self.results(y_test, y_pred)
            
            