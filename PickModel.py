from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
import numpy as np
import sys

class PickModel():
    
    def __init__(self, network_name, use_pretrained = True, num_classes = 50):
        super(PickModel, self).__init__()
        self.network_name = network_name
        self.use_pretrained = use_pretrained
        self.num_classes = num_classes
        
    def get_model(self):
        model_name = self.network_name
        if model_name == "Resnet18":
            print("[!] Using Resnet18 model")
            self.model_ft = models.resnet18(pretrained=self.use_pretrained)
            #self.set_parameter_requires_grad(self.model_ft, self.feature_extract)
            self.num_ftrs = self.model_ft.fc.in_features
            self.model_ft.fc = nn.Linear(self.num_ftrs, self.num_classes)
            self.input_size = 244
               
        elif model_name == "Resnet101":
            print("[!] Using Resnet101 model")
            self.model_ft = models.resnet101(pretrained=self.use_pretrained)
            #self.set_parameter_requires_grad(self.model_ft, self.feature_extract)
            self.num_of_features = self.model_ft.fc.in_features
            self.model_ft.fc = nn.Linear(self.num_of_features, self.num_classes)
            self.input_size = 244
            
        elif model_name == "Resnet50":
            print("[!] Using Resnet50 model")
            self.model_ft = models.resnet50(pretrained=self.use_pretrained)
            #self.set_parameter_requires_grad(self.model_ft, self.feature_extract)
            self.num_of_features = self.model_ft.fc.in_features
            self.model_ft.fc = nn.Linear(self.num_of_features, self.num_classes)
            self.input_size = 244
        
        elif model_name == "Densenet169":
            print("[!] Using Densenet169 model")
            self.model_ft = models.densenet169(pretrained=self.use_pretrained)
            #self.set_parameter_requires_grad(self.model_ft, self.feature_extract)
            self.model_ft.classifier = (nn.Linear(1664, self.num_classes))
            self.input_size = 244
            
        elif model_name == "Densenet121":
            print("[!] Using Densenet121 model")
            self.model_ft = models.densenet121(pretrained=self.use_pretrained)
            #self.set_parameter_requires_grad(self.model_ft, self.feature_extract)
            self.model_ft.classifier = (nn.Linear(1024, self.num_classes))
            self.input_size = 244
        
        elif model_name == "Densenet201":
            print("[!] Using Densenet201 model")
            self.model_ft = models.densenet201(pretrained=self.use_pretrained, drop_rate = self.drop_rate)
            #self.set_parameter_requires_grad(self.model_ft, self.feature_extract)
            self.model_ft.classifier = (nn.Linear(1920, self.num_classes))
            self.input_size = 244
            
        elif model_name == "Densenet161":
            print("[!] Using Densenet161 model")
            self.model_ft = models.densenet161(pretrained=self.use_pretrained)
            #self.set_parameter_requires_grad(self.model_ft, self.feature_extract)
            self.model_ft.classifier = (nn.Linear(2208, self.num_classes))
            self.input_size = 244
        
        elif model_name == "DiatomsNetwork":
            print("[!] DiatomsNetwork model")
            self.model_ft = DiatomsNetwork(self.num_classes)
            self.input_size = 244
                    
        elif model_name == "SqueezeNet":
            print("[!] SqueezeNet model")
            self.model_ft = models.squeezenet1_1(pretrained=self.use_pretrained)
            self.in_ftrs = self.model_ft.classifier[1].in_channels
            self.out_ftrs = self.model_ft.classifier[1].out_channels
            features = list(self.model_ft.classifier.children())
            features[1] = nn.Conv2d(self.in_ftrs, self.num_classes,1,1)
            features[3] = nn.AvgPool2d(13,stride=1)
            
            self.model_ft.classifier = nn.Sequential(*features)
            self.model_ft.num_classes = self.num_classes
            self.input_size = 244
            
        else:
            print("[x] Invalid model name, exiting!")
            sys.exit()
            
        return self.model_ft