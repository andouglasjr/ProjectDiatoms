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
np.set_printoptions(threshold=np.nan)
    
def create_folders_to_results(args, params):
    ################################################################################################################
    #Folder names to save the log and model
    ################################################################################################################
    localtime = args.time_training
    lr = params['lr']
    folder_save_results_epoch = args.save_dir+'/'+args.network_name+'/lr_'+str(lr)+'_'+str(localtime)+'/epochs'

    if not os.path.exists(folder_save_results_epoch):
        os.makedirs(folder_save_results_epoch)

    folder_best_result = args.save_dir + '/'+args.network_name+'/lr_'+ str(lr)+'_'+str(localtime)+'/best_result.pt'
    return folder_save_results_epoch, folder_best_result
    ################################################################################################################
    
        
def train(args, device):
    data_log.log("Starting training", 'l')
    
    #Get args and put it in variables
    network_name = args.network_name
    loss_function = args.loss_function
    
    #Get dataset
    dataset = DataUtils(device = device, args = args)
    dataloaders = dataset.load_data()    
    train_size, val_size = dataset.get_dataset_sizes()
    data_log.log("DataSet Size (Train: {}, Validation: {})".format(train_size, val_size), 'v')

    for count in range(args.range):  
        lr_center_loss = 10**random.uniform(0,-1)
        lr = utils.get_learning_rate(args, network_name)
        #lr_center_loss = 0.9271141509078993

        params = {
            'lr' : lr,
            'momentum' : 0.9,
            'step_size' : 5,
            'gamma' : 0.1,
            'set_criterion' : True,
            'num_epochs' : args.epochs,
            'net_name' : network_name,
            'loss_function': loss_function,
            'lr_center_loss': lr_center_loss
        }

        data_log.log("Network Architeture: {}".format(network_name), 'l')
        data_log.log("Parameters:", 'l')
        data_log.log("Number of Epochs: {}".format(params['num_epochs']), 'e')
        data_log.log("Learning Rate: {}".format(params['lr']), 'e')
        data_log.log("Momentum: {}".format(params['momentum']), 'e')
        data_log.log("Gamma: {}".format(params['gamma']), 'e')
        data_log.log("LR Center Loss: {}".format(lr_center_loss), 'e')

        #Create results forlder
        folder_save_results_epoch, folder_best_result = create_folders_to_results(args, params)

        ################################################################################################################
        #Training
        ################################################################################################################
        model_ft = ModelClass(model_name=network_name, num_classes = int(args.classes_training), log = data_log)
        model = model_ft.get_model()
        best_model = model_ft.train_model(model, dataloaders, params, dataset, args)
        model_ft.save_model(best_model, folder_best_result)
        ################################################################################################################


        ################################################################################################################
        #Analysis
        ################################################################################################################
        test(args, device, best_model)

def test(args, device, model):
    if args.weights is None:
        print('No weights are provided. Will test using random initialized weights.')
    data_log.log("Analyzing Results to {}".format(args.network_name), 'l')

    folder_name = ['test_diatoms_3_class']
    data_test = DataUtils(folder_names = folder_name, device = device, args = args)

    dataloaders_test = data_test.load_data(dataset_name = 'test')


    dataset_size_test = len(data_test.images_dataset)
    data_log.log("DataSet Size (Test: {})".format(dataset_size_test), 'v')
    results,correct,incorrect,image_incorrect, correct_class = ModelClass.test_model(model, dataloaders_test, list_of_name_folders[1], data_test, device, data_log, args)



    data_test.save_results(results,correct,incorrect, correct_class, data_log, image_incorrect, True)      

    
if __name__ == "__main__":
    ################################################################################################################
    #Setup
    ################################################################################################################
    
    args = ArgumentsParser().get_args()    
    #Init Log
    data_log = DataLogger(args)
    data_log.log("Diatoms Project - ISASI/Natalnet", 'l')
    list_of_name_folders = ['Diatom50NEW_generated', 'test_diatoms_3_class']
    data_dir = args.data_dir
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    
    ################################################################################################################
    # Train or test?
    ################################################################################################################
    
    if args.weights is not None:  # init the model weights with provided one
        best_model = torch.load(args.weights)
    if not args.testing:
        train(args, device)
    else:
        test(args, device, best_model)
        
    data_log.log("Close Log", 'l')
    
    
    
    