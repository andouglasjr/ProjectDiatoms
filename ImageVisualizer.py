######################
# Title: 
# Note:
# Author:
######################
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import torch

class ImageVisualizer:
    def __init__(self, folders_names, mean, std):
        self.folders_names = folders_names
        self.mean = mean
        self.std = std
        
    def imshow(self, inputs, title=None, isToShow=True, folder_name=None):
        """Imshow for Tensor."""
        inputs = inputs.numpy().transpose((1, 2, 0))
        if folder_name is not None:
            inputs = np.array(self.std[folder_name]) * inputs + np.array(self.mean[folder_name])
        inputs = np.clip(inputs, 0, 1)
        if isToShow:
            plt.imshow(inputs)
        else:
            plt.savefig('result.png')
        
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated
    
    def set_normalization(self, mean, std):
        self.mean = mean
        self.std = std
    
    def get_normalization(self, image_visualizer):
        return image_visualizer.mean, image_visualizer.std
        
    def show_one_image_PIL(self, inputs=None, title=None, isToShow=True, folder_name=None):
        centercrop = transforms.CenterCrop(224)
        totensor = transforms.ToTensor()
        mean, std = self.get_normalization(self)
        normalize = transforms.Normalize(mean[folder_name], std[folder_name])
        
        img = centercrop(inputs)
        img = totensor(img)
        img = normalize(img)
        self.imshow(img, title, isToShow, folder_name)
        
    def show_one_image_tensor(self, inputs=None, title=None, isToShow=True, folder_name=None):
        self.imshow(inputs, title, isToShow, folder_name)
    
    def visualize_model(self, model, dataloaders, folder_name, num_images=6):
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders[folder_name]):
                #inputs = inputs.to(device)
                #labels = labels.to(device)
                labels = [int(class_names[l.item()]) for l in labels]

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 4, images_so_far)
                    ax.axis('off')
                    ax.set_title('predicted: {}'.format(preds[j]))
                    imshow(inputs.cpu().data[j])

                    images_so_far += 1

                    for t, (inp, lab) in enumerate(dataloaders_per_class[folder_name]):
                        #print(labels[j])
                        if(labels[j] == lab):
                            ax = plt.subplot(num_images//2, 4, images_so_far)
                            ax.axis('off')
                            ax.set_title('class: {}'.format(lab[0]))
                            imshow(inp.cpu().data[0])
                            if (images_so_far == num_images):
                                model.train(mode=was_training)
                                return
                            break
            model.train(mode=was_training)
        
        
        