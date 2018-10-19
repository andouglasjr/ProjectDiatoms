from DataUtils import DataUtils
from ModelClass import ModelClass
from torchvision import datasets, models, transforms
import os
list_of_name_folders = ['train_diatoms_3_class_simulate_1','val_diatoms_3_class_simulate_1']
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
        transforms.Normalize([0.5017087, 0.5017087, 0.5017087], [0.08929635, 0.08929635, 0.08929635])
        #transforms.Normalize([0.493], [0.085])
    ]),
    list_of_name_folders[1]: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        #transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize([0.5017087, 0.5017087, 0.5017087], [0.08929635, 0.08929635, 0.08929635])
        #transforms.Normalize([0.496], [0.07])
    ])
}

#------------------------------------------------
#Data Loaders 
#------------------------------------------------
#list_of_name_folders = ['train_diatoms_3_class','val_diatoms_3_class']
data_dir = '../data'

model_name='Resnet50'

test_names = ['Resnet18','Resnet50']

for t in test_names:

    data = DataUtils(list_of_name_folders, data_dir, data_transforms, net_name = t)
    image_datasets = data.get_all_image_datasets()
    dataloaders = data.load_data(image_datasets)
    dataset_size = data.get_dataset_size()

    #Prameters of training
    params = {
        'lr' : 0.001,
        'momentum' : 0.9,
        'step_size' : 4,
        'gamma' : 0.00001,
        'set_criterion' : True,
        'num_epochs' : 8
    }


    model_ft = ModelClass(model_name=t, folder_names = list_of_name_folders)
    model = model_ft.get_model()
    best_model = model_ft.train_model(model, dataloaders, params, dataset_size, data)
    model_ft.save_model(best_model, 'results/' + t + '.pt')

