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
import numpy as np
np.set_printoptions(threshold=np.nan)

def setup(args):
    #Init Log
    data_log = DataLogger(args)
    data_log.log("Starting code...", 'l')
    
    list_of_name_folders = ['Diatom50NEW_generated', 'test_diatoms_3_class']
    #list_of_name_folders = ['train_diatoms_3_class_simulate_all', 'test_diatoms_3_class']
    #list_of_name_folders = ['train_diatoms_3_class_simulate_1','val_diatoms_3_class_simulate_1', 'test_diatoms_3_class']
    #list_of_name_folders = ['test_diatoms_3_class','test_diatoms_3_class_simulate','test_diatoms_3_class']

    data_transforms_to_compute_mean = {
        list_of_name_folders[0]: transforms.Compose([
            transforms.ToTensor(),
        ]),
        list_of_name_folders[1]: transforms.Compose([
            transforms.ToTensor(),
        ])
    }


    #------------------------------------------------
    #Data Loaders 
    #------------------------------------------------
    data_dir = args.data_dir
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #Computing the Mean and Std of trains dataset
    #data_log.log("Computing dataset train mean and std...", 'l')
    #data_mean = DataUtils(list_of_name_folders, data_dir, data_transforms_to_compute_mean, net_name = '', device = device)
    #image_datasets_mean = data_mean.get_image_datasets()
    #mean, std = data_mean.compute_mean(image_datasets_mean, list_of_name_folders[0])
    mean, std = [0.5018, 0.5018, 0.5018],[0.0837, 0.0837, 0.0837]
    data_log.log("Mean: {}, Std: {}".format(mean, std), 'v')

    #Tranformation for Trainning
    #------------------------------------------------
    #Data Augmentation
    #------------------------------------------------
    data_transforms = {
        list_of_name_folders[0]: transforms.Compose([
            #transforms.Resize(256),
            transforms.CenterCrop(224),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(180),
            #transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            #transforms.Normalize([0.493], [0.085])
        ]),
        list_of_name_folders[1]: transforms.Compose([
            #transforms.Resize(256),
            transforms.CenterCrop(224),
            #transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            #transforms.Normalize([0.496], [0.07])
        ])
    }
    
    return data_log, list_of_name_folders, data_transforms, device
    

def get_learning_rate(args, network_name):
    if args.exponential_range is not None:
        return 10**random.uniform(args.exponential_range[0], args.exponential_range[1])
    if args.new_lr:
        return args.lr
    else:
        if(network_name == "Resnet50"):
            return 3.118464108103618e-05
        elif(network_name == "Densenet201"):
            return 8.832537199285954e-04
        elif(network_name == "Densenet169"):
            return 2.6909353460670058e-05
        
        
def train(args):
    data_log, list_of_name_folders, data_transforms, device = setup(args)
    
    for network_name in args.network_name:
        
            # train or test
        if args.weights is not None:  # init the model weights with provided one
            #best_model = model.load_state_dict(torch.load(args.weights))
            best_model = torch.load(args.weights)
        if not args.testing:
            data = DataUtils(device = device, args = args)
            data_log.log("Starting training", 'l')
            dataloaders = data.load_data()
            dataset_size = data.get_dataset_size()

            train_size = dataset_size * 0.8
            val_size = dataset_size * 0.2

            data_log.log("DataSet Size (Train: {}, Validation: {})".format(train_size, val_size), 'v')

            best_accuracy_old = 0
            drop_rate = 0
            count = 0

            loss_function = args.loss_function

            for count in range(args.range):  
                lr_center_loss = 10**random.uniform(0,-1)
                lr = get_learning_rate(args, network_name)
                #lr_center_loss = 0.9271141509078993

                params = {
                    'lr' : lr,
                    'momentum' : args.lr_decay,
                    'step_size' : 1,
                    'gamma' : 0.1,
                    'set_criterion' : True,
                    'num_epochs' : args.epochs,
                    'net_name' : network_name,
                    'drop_rate' : drop_rate,
                    'loss_function': loss_function,
                    'lr_center_loss': lr_center_loss
                }

                data_log.log("Network Architeture: {}".format(network_name), 'l')
                data_log.log("Parameters:", 'l')
                data_log.log("Number of Epochs: {}".format(params['num_epochs']), 'e')
                data_log.log("Learning Rate: {}".format(params['lr']), 'e')
                data_log.log("Drop Rate: {}".format(drop_rate), 'e')
                data_log.log("Momentum: {}".format(params['momentum']), 'e')
                data_log.log("Gamma: {}".format(params['gamma']), 'e')
                data_log.log("LR Center Loss: {}".format(lr_center_loss), 'e')

                folder_save_results_epoch = args.save_dir+'/'+network_name+'/lr_'+str(lr)+'/epochs'

                if not os.path.exists(folder_save_results_epoch):
                    os.makedirs(folder_save_results_epoch)

                folder_best_result = args.save_dir + '/'+network_name+'/lr_'+ str(lr)+'/best_result.pt'


                feature_extract=False
                #num_of_layers=count*6+4 #Densenet
                num_of_layers=8 #Resnet

                #print(count)

                model_ft = ModelClass(model_name=network_name, num_classes = 50, feature_extract=feature_extract, num_of_layers=num_of_layers, folder_names = ['train','val'], log = data_log, drop_rate = drop_rate)
                model = model_ft.get_model()
                #print(model)

                best_model = model_ft.train_model(model, dataloaders, params, data, args)
                model_ft.save_model(best_model, folder_best_result)
                
                data_test = DataUtils(device = device, args = args)
                dataloaders_test = data_test.load_data(dataset_name = 'test')
                dataset_size_test = data_test.get_dataset_size()
                
                data_log.log("Analyzing Results to {}".format(network_name), 'l')
                results,correct,incorrect,image_incorrect, correct_class = model_ft.test_model(best_model, dataloaders_test, list_of_name_folders[1],data, args)
                data.save_results(results,correct,incorrect, correct_class, data_log, image_incorrect)
                
        else:  # testing
            if args.weights is None:
                print('No weights are provided. Will test using random initialized weights.')
            data_log.log("Analyzing Results to {}".format(network_name), 'l')
            
            folder_name = ['test_diatoms_3_class']
            data_test = DataUtils(folder_names = folder_name, device = device, args = args)
            dataloaders_test = data_test.load_data(dataset_name = 'test')
            dataset_size_test = len(data_test.images_dataset)
            data_log.log("DataSet Size (Test: {})".format(dataset_size_test), 'v')
            
            results,correct,incorrect,image_incorrect, correct_class = ModelClass.test_model(best_model, dataloaders_test, list_of_name_folders[1], data_test, device, data_log, args)
            data_test.save_results(results,correct,incorrect, correct_class, data_log, image_incorrect, True)
            #show_reconstruction(best_model, dataloaders[list_of_name_folders[2]], 12, args, network_name)
                
            
            #Visualizing Results
            #visual = ImageVisualizer(list_of_name_folders, mean, std)
            #visual.call_visualize_misclassifications(correct_class, visual, image_incorrect)
        data_log.log("Close Log", 'l')
    
if __name__ == "__main__":
    import argparse
    import os
    
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Diatoms Research CNR-ISASI")
    parser.add_argument('--network_name', nargs='+',
                       help="Insert the network name for trainning, e.g. --network_name Resnet18 : Could be one list")
    parser.add_argument('--new_lr', action='store_true',
                       help="If not setted will be used the lr already found!")
    parser.add_argument('--range', default=1, type=int,
                       help="How much times the network will be trainned with diferent learning rates!")
    parser.add_argument('--exponential_range', nargs='*', default=None, type=float,
                       help="What is the exponential range to find the best lr, e.g. --exponential_range -3 -5")
    parser.add_argument('--epochs', default=8, type=int,
                       help="Number of epochs")
    parser.add_argument('--batch_size', default=256, type=int,
                       help="Size of the batch")
    parser.add_argument('--lr', default=0.0001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--loss_function', default = 'cross_entropy',
                       help="Which will be loss function used? Cross Validation (default), Center Loss or both")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--data_dir', default='../data/Dataset_5/Diatom50NEW_generated',
                        help="Directory of data. If no data, use \'--download\' flag to download it")
    parser.add_argument('--save_dir', default='results',
                       help="Directory to save the results!")
    parser.add_argument('--dataset', default='results',
                       help="Directory where is the data!")
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    parser.add_argument('--plot', action='store_true', help="whether to plot features for every epoch")
    parser.add_argument('--images_per_class', default=21000, help="how many images will be used per class")
    parser.add_argument('--classes_training', default=50, help="how many classes there are in the training")
    
    args = parser.parse_args()
    print(args)
    
    #for net_name in args.network_name:
    #    if not os.path.exists(args.save_dir+'/'+net_name+'/'+/):
    #        os.makedirs(args.save_dir+'/'+net_name)
    
    train(args)


    
    
    
    