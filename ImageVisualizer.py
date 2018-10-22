######################
# Title: 
# Note:
# Author:
######################
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from DiatomsDataset import DiatomsDataset

class ImageVisualizer:
    def __init__(self, folders_names, mean, std):
        self.folders_names = folders_names
        self.mean = mean
        self.std = std
        
        self.data_transforms = {
            'train_': transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])
        }
        
        self.image_per_class ={x: datasets.ImageFolder(os.path.join('../data', x), self.data_transforms[x]) for x in ['train_']}
        self.dataloaders_per_class = {x: torch.utils.data.DataLoader(self.image_per_class[x], shuffle=False, batch_size=50, num_workers=1) for x in ['train_']}
 
    def imshow(self, inputs, title=None, isToShow=True, folder_name=None):
        """Imshow for Tensor."""
        inputs = inputs.numpy().transpose((1, 2, 0))
        if folder_name is not None:
            inputs = np.array(self.std) * inputs + np.array(self.mean)
        inputs = np.clip(inputs, 0, 1)
        if isToShow:
            plt.imshow(inputs)
            
        else:
            plt.savefig('result.png')
        
        if title is not None:
            plt.title(title)
        plt.pause(0.01)  # pause a bit so that plots are updated
        
    
    def set_normalization(self, mean, std):
        self.mean = mean
        self.std = std
    
    def get_normalization(self, image_visualizer):
        return image_visualizer.mean, image_visualizer.std
        
    def show_one_image_PIL(self, inputs=None, title=None, isToShow=True, folder_name=None):
        centercrop = transforms.CenterCrop(224)
        totensor = transforms.ToTensor()
        mean, std = self.get_normalization(self)
        normalize = transforms.Normalize(mean, std)
        
        img = centercrop(inputs)
        img = totensor(img)
        img = normalize(img)
        self.imshow(img, title, isToShow, folder_name)
        
    def show_one_image_tensor(self, inputs=None, title=None, isToShow=True, folder_name=None):
        fig = plt.figure()
        self.imshow(inputs, title, isToShow, folder_name)
        
    
    def visualize_model(self, model, dataloaders, folder_name, image_datasets, num_images=6):
        was_training = model.training
        model.eval()
        images_so_far = 0
        
        class_names = image_datasets[folder_name].classes
        
        diatoms_class = DiatomsDataset('train_', '../data', self.data_transforms)

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders[folder_name]):
                #inputs = inputs.to(device)
                #labels = labels.to(device)
                labels = [int(class_names[l.item()]) for l in labels]
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                preds_names = [int(class_names[l.item()]) for l in preds]
    
                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 4, images_so_far)
                    ax.axis('off')
                    ax.set_title('predicted: {}'.format(preds_names[j]))
                    self.imshow(inputs.data[j], folder_name = folder_name)

                    images_so_far += 1
                    
                    for k in range(50):
                        if(labels[j] == diatoms_class[k]['diatoms']):
                            #print(lab[k])
                            ax = plt.subplot(num_images//2, 4, images_so_far)
                            ax.axis('off')
                            ax.set_title('class: {}'.format(diatoms_class[k]['diatoms']))
                            #self.imshow(inp.data[k-1])
                            self.imshow(diatoms_class[k]['image'])
                            if (images_so_far == num_images):
                                model.train(mode=was_training)
                                return
                            break
            model.train(mode=was_training)
        
    def visualize_misclassification(self, predicts_wrong, class_correct):
        images_so_far = 0
        
        fig = plt.figure()
        #class_names = image_datasets[folder_name].classes
        #class_correct = [int(class_names[l.item()]) for l in labels]
        
        diatoms_class = DiatomsDataset('train_', '../data', self.data_transforms)
        for i in range(len(diatoms_class)):
            if(diatoms_class[i]['diatoms'] == class_correct):
                img_class_correct = diatoms_class[i]['image']
                diatoms_class_correct = diatoms_class[i]['diatoms']
                break

        num_images = len(predicts_wrong)
        #print(num_images)

        images_so_far += 1
        ax = plt.subplot(num_images//2, 4, images_so_far)
        ax.axis('off')
        ax.set_title('Correct Class: {}'.format(diatoms_class_correct))
        self.imshow(img_class_correct)

        for j in range(1,num_images):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 4, images_so_far)
            ax.axis('off')
            ax.set_title('Predict: {}'.format(predicts_wrong[j]['class']))
            self.imshow(predicts_wrong[j]['image'], folder_name = 'train_')

        
        