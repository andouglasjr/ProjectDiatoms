import os
import torch
from DataUtils import DataUtils
from ModelClass import ModelClass
from torchvision import datasets, models, transforms
from DataLogger import DataLogger
from ImageVisualizer import ImageVisualizer
from torch.utils import data as D
import utils as utils
import matplotlib.pyplot as plt
import matplotlib
import random
import torch.nn as nn
import numpy as np
import time
import torchvision
import glob
import os.path as osp
from PIL import Image
import argparse
from ArgumentsParser import ArgumentsParser
from sklearn.externals import joblib
from sklearn import model_selection
import xgboost
from PickModel import PickModel
from TrainingClass import TrainingClass
from ModelUtils import ModelUtils
from Ensemble import Ensemble
from TestingClass import TestingClass

np.set_printoptions(threshold=np.nan)
          
def train(args, device):
    data_log.log("Starting training", 'l')
    
    #Get args and put it in variables
    network_name = args.network_name
    loss_function = args.loss_function
    
    #Get dataset
    dataset = DataUtils(device = device, args = args)
    train_size, val_size = dataset.get_dataset_sizes()
    data_log.log("DataSet Size (Train: {}, Validation: {})".format(train_size, val_size), 'v')

    for count in range(args.range):  
        args.lr = utils.get_learning_rate(args, network_name)
        data_log.log("Network Architeture: {}".format(network_name), 'l')
        data_log.log("Parameters:", 'l')
        data_log.log("Number of Epochs: {}".format(args.epochs), 'e')
        data_log.log("Learning Rate: {}".format(args.lr), 'e')
        data_log.log("Momentum: {}".format(args.momentum), 'e')
        data_log.log("Gamma: {}".format(args.gamma), 'e')

        #Create results folder
        folder_save_results_epoch, folder_best_result = utils.create_folders_to_results(args)

        ################################################################################################################
        #Training
        ################################################################################################################
        #model_ft = ModelClass(model_name=network_name, num_classes = int(args.classes_training), log = data_log)
        #model = model_ft.get_model()
        #best_model = model_ft.train_model(model, dataloaders, params, dataset, args)
        #model_ft.save_model(best_model, folder_best_result)
        #model_1 = PickModel("Resnet50", num_classes = int(args.classes_training)).get_model()
        #model_2 = PickModel("Resnet101", num_classes = int(args.classes_training)).get_model()            
        model_1 = torch.load("results/Resnet50/lr_0.0003118464108103618_Mon Feb 25 20:01:50 2019/epochs/epoch_15.pt")
        model_2 = torch.load("results/Resnet101/lr_0.0003118464108103618_Fri Feb 22 14:07:07 2019/epochs/epoch_5.pt")
        models = []
        models.append(model_1)
        models.append(model_2)
        #ModelUtils.set_parameter_requires_grad(model_1.module, True, 20)
        #ModelUtils.set_parameter_requires_grad(model_2.module, True, 20)
        
        #test(args, device, model_1)
        #test(args, device, model_2)
                             
        #train_model = Ensemble(model_1.module, model_2.module)
        #print(train_model)
        best_model = TrainingClass(model_1, dataset, args).fit_ensemble(models)
        
        #ModelUtils.save_model(best_model, folder_best_result)

        ################################################################################################################
        #Analysis
        ################################################################################################################
        test(args, device, best_model)

def test(args, device, model):
    if args.weights is None:
        print('No weights are provided. Will test using random initialized weights.')
    data_log.log("Analyzing Results to {}".format(args.network_name), 'l')

    folder_name = ['test_diatoms_3_class']
    #folder_name = ['Diatom50NEW_focus']
    data_test = DataUtils(folder_names = folder_name, device = device, args = args)
    dataset_size_test = len(data_test.images_dataset)
    data_log.log("DataSet Size (Test: {})".format(dataset_size_test), 'v')
    
    ##############################################################################
    #This srcipt part tests using more than one pre-trained model
    #results,correct,incorrect,image_incorrect, correct_class = ModelClass.test_models(model, dataloaders_test, list_of_name_folders[1], data_test, device, data_log, args)
    ##############################################################################
    
    ##############################################################################
    #This srcipt part tests using one model pre-trained based on weights parameter
    
    #TestingClass(model, data_test, args).test()
    model_1 = torch.load("results/Resnet50/lr_0.0003118464108103618_Mon Feb 25 20:01:50 2019/epochs/epoch_15.pt")
    model_2 = torch.load("results/Resnet101/lr_0.0003118464108103618_Fri Feb 22 14:07:07 2019/epochs/epoch_5.pt")
    models = []
    models.append(model_1)
    models.append(model_2)
    TestingClass(model, data_test, args).test_ensemble(models)
    
    ##############################################################################
    
    ##############################################################################
    #This srcipt part tests using the xgb model
    
    #xgb_model = joblib.load("pima.pickle.dat")
    #ModelClass.test_xgboost_model(model, xgb_model, dataloaders_test, list_of_name_folders[1], data_test, device, data_log, args)
    ##############################################################################

    ##############################################################################
    #This srcipt part tests using the votting algorithm
    #ModelClass.test_models_votting(model, dataloaders_test, list_of_name_folders[1], data_test, device, data_log, args)

    #data_test.save_results(results,correct,incorrect, correct_class, data_log, image_incorrect, True)      

    
if __name__ == "__main__":
    ################################################################################################################
    #Setup
    ################################################################################################################
    args = ArgumentsParser().get_args()    
    
    #Init Log
    data_log = DataLogger(args).getInstance(args)
    data_log.log("Diatoms Project - ISASI/Natalnet", 'l')
    #list_of_name_folders = ['Diatom50NEW_generated', 'test_diatoms_3_class']
    list_of_name_folders = ['Diatom50NEW_generated', 'Diatom50NEW_focus']

    data_dir = args.data_dir
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
        
    if args.weights is not None:  # init the model weights with provided one
        best_model = torch.load(args.weights)
        #model_names = ["results/Resnet50/lr_0.0003118464108103618_Mon Feb 25 20:01:50 2019/epochs/epoch_15.pt", 
        #              "results/Resnet101/lr_0.0003118464108103618_Fri Feb 22 14:07:07 2019/epochs/epoch_5.pt"]
                      #"results/Resnet101/lr_0.0003118464108103618_Thu Feb 21 11:01:00 2019/epochs/epoch_3.pt"]
        #best_model = []
        
        #best_model.append(torch.load(model_names[0]))
        #best_model.append(torch.load(model_names[1]))
                       
        
        #best_model.append(torch.load(model_names[2]))
        
    if not args.testing:
        train(args, device)
    else:
        test(args, device, best_model)
        
    data_log.log("Close Log", 'l')
    
    
    
    