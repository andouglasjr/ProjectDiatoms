import os
import torch
from DataUtils import DataUtils
from ModelClass import ModelClass
from torchvision import datasets, models, transforms
from DataLogger import DataLogger
from ImageVisualizer import ImageVisualizer
import matplotlib.pyplot as plt
import matplotlib
import random
import torch.nn as nn

#Init Log
data_log = DataLogger()
data_log.log("Init training code...", 'l')
#plt.ion()
list_of_name_folders = ['train_diatoms_3_class_simulate_all','val_diatoms_3_class_simulate_all', 'test_diatoms_3_class']
#list_of_name_folders = ['train_diatoms_3_class_simulate_1','val_diatoms_3_class_simulate_1', 'test_diatoms_3_class']
#list_of_name_folders = ['test_diatoms_3_class','test_diatoms_3_class_simulate','test_diatoms_3_class']

data_transforms_to_compute_mean = {
    list_of_name_folders[0]: transforms.Compose([
        transforms.ToTensor(),
    ]),
    list_of_name_folders[1]: transforms.Compose([
        transforms.ToTensor(),
    ]),
    list_of_name_folders[2]: transforms.Compose([
        transforms.ToTensor(),
    ])
}


#------------------------------------------------
#Data Loaders 
#------------------------------------------------
#list_of_name_folders = ['train_diatoms_3_class','val_diatoms_3_class']
data_dir = '../data'

model_name='Resnet50'
test_names = ['Resnet18','Resnet50']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_log.log("Computing dataset train mean and std...", 'l')
#Computing the Mean and Std of trains dataset
data_mean = DataUtils(list_of_name_folders, data_dir, data_transforms_to_compute_mean, net_name = '', device = device)
image_datasets_mean = data_mean.get_all_image_datasets()
#mean, std = data_mean.compute_mean(image_datasets_mean, list_of_name_folders[0])
mean, std = [0.5018, 0.5018, 0.5018],[0.0837, 0.0837, 0.0837]
data_log.log("Mean: {}, Std: {}".format(mean, std), 'v')

#Tranformation for Trainning
#------------------------------------------------
#Data Augmentation
#------------------------------------------------
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

data_log.log("Starting training", 'l')
#for t in test_names:
for t in ['Densenet169']:
#for t in ['Resnet50']:

    data = DataUtils(list_of_name_folders, data_dir, data_transforms, net_name = t, device = device)
    image_datasets = data.get_all_image_datasets()
    dataloaders = data.load_data(image_datasets)
    dataset_size = data.get_dataset_size()
    
    data_log.log("DataSet Size: {}".format(dataset_size), 'v')
    best_accuracy_old = 0
    
    for count in range(1):
    #for num_of_layers in range(1):
        #lr = 3.118464108103618e-05 #Learning rate to Resnet50
        #lr =10**random.uniform(-4,-5)
        #lr =3.469783971737552e-05
        lr = 2.6909353460670058e-05 #Learning rate to Densenet169
        #Prameters of training
        #best lr=0.0005
        params = {
            'lr' : lr,
            'momentum' : 0.9,
            'step_size' : 4,
            'gamma' : 0.00001,
            'set_criterion' : True,
            'num_epochs' : 6,
            'net_name' : t
        }

        data_log.log("Network Architeture: {}".format(t), 'l')
        data_log.log("Parameters:", 'l')
        data_log.log("Number of Epochs: {}".format(params['num_epochs']), 'e')
        data_log.log("Learning Rate: {}".format(params['lr']), 'e')

        feature_extract=False 
        num_of_layers=count*6+4
        #print(count)
        
        model_ft = ModelClass(model_name=t,channels=3, feature_extract=feature_extract, num_of_layers=num_of_layers, folder_names = list_of_name_folders, log = data_log)
        model = model_ft.get_model()
        #print(model)
        #best_model = model_ft.train_model(model, dataloaders, params, dataset_size, data)
        #model_ft.save_model(best_model, 'results/all_' + t + '_' +str(lr)+'.pt')
    
        #Analyzing Results
        #data_log.log("Analyzing Results to {}".format(t), 'l')
        best_model = model_ft.load_model('results/BestResultDensenet_1/all_Densenet169_2.6909353460670058e-05.pt', '')
        #best_model = model_ft.load_model('results/all_'+t+'_'+ str(lr)+ '.pt', '')
        #best_model = model_ft.load_model('results/Resnet50.pt', 'cpu')

        #Visualizing Results
        visual = ImageVisualizer(list_of_name_folders, mean, std)
        results,correct,incorrect,image_incorrect, correct_class = model_ft.confusion_matrix(best_model, dataloaders, list_of_name_folders[2],data)

        data.save_results(results,correct,incorrect, correct_class, data_log)
        

        visual.call_visualize_misclassifications(correct_class, visual, image_incorrect)
    data_log.log("Close Log", 'l')
