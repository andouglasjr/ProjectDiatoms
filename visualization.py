from DataUtils import DataUtils 
from ModelClass import ModelClass
from DataLogger import DataLogger
from ArgumentsParser import ArgumentsParser
from PIL import Image
import matplotlib.pyplot as plt
import torchvision

import torch

args = ArgumentsParser().get_args()
#Init Log
data_log = DataLogger(args)
#data_log.log("Diatoms Project - ISASI/Natalnet", 'l')
list_of_name_folders = ['train_', 'test_diatoms_3_class']
data_dir = args.data_dir
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    

#Get args and put it in variables
network_name = args.network_name
loss_function = args.loss_function

#Get dataset
dataset = DataUtils(device = device,   args = args, folder_names = list_of_name_folders)
dataloaders = dataset.load_data(validation_split = 0)

model_ft = ModelClass(model_name=network_name, num_classes = int(args.classes_training), log = data_log)
model = model_ft.get_model()

#new_features = torch.nn.Sequential(*list(model.children()))
#for param in new_features[0]:
#p =[]
#cont = 0
#for param in model.parameters():
#    p.append(param)
#    print(cont)
#    cont = cont + 1
#print(p[159].shape)
#plt.figure(figsize=(16,8))
#grid_image = torchvision.utils.make_grid(p[159], nrow=8)
#print(grid_image.shape)
#grid_image = grid_image.permute(1,2,0).detach().numpy()
#plt.imshow(grid_image)  

best_model = torch.load("/home/andouglas/Desktop/Results/all_lr_0.0001_drop_0.pt", map_location='cpu')
p =[]
cont = 0
for name, param in best_model.named_parameters():
    p.append(param)
    print(param.shape, name, cont)
    cont=cont+1
print(p[153].shape)
plt.figure(figsize=(16,8))
grid_image = torchvision.utils.make_grid(p[0], nrow=8)
print(grid_image.shape)
grid_image = grid_image.permute(1,2,0).detach().numpy()
plt.imshow(grid_image)  
plt.show() 




