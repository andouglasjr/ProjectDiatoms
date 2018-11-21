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
from sklearn.metrics import f1_score
from CenterLoss import CenterLoss
import keyboard

class ModelClass():
    
    def __init__(self, model_name="", channels = 3, num_classes = 3, feature_extract=False, num_of_layers=0, use_pretrained=True, folder_names = None, device = None, log = None, drop_rate = 0):
        self.model_name = model_name
        self.num_classes = num_classes
        self.feature_extract = feature_extract
        self.use_pretrained = use_pretrained
        self.num_of_layers = num_of_layers
        self.drop_rate = drop_rate
        if(device == None):
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        if (folder_names == None):
            self.folder_names = ['train_diatoms_3_class','val_diatoms_3_class']
        else:
            self.folder_names = folder_names
            
        self.channels = channels
        self.model_ft = None
        self.log = log
        input_size = 0
        self.best_loss = 1000
        self.cont_to_stop = 0
        self.num_of_features = 0
        
        if model_name == "Resnet18":
            print("[!] Using Resnet18 model")
            self.model_ft = models.resnet18(pretrained=use_pretrained)
            self.set_parameter_requires_grad(self.model_ft, self.feature_extract)
            num_ftrs = self.model_ft.fc.in_features
            self.model_ft.fc = nn.Linear(num_ftrs, num_classes)
            if (self.channels == 1):
                new_features = nn.Sequential(*list(self.model_ft.children()))
                pretrained_weights = new_features[0].weight
                new_features[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2)
                print(new_features[0])
                # For M-channel weight should randomly initialized with Gaussian
                new_features[0].weight.data.normal_(0, 0.001)
                # For RGB it should be copied from pretrained weights
                #new_features[0].weight.data[:, :3, :, :] = pretrained_weights
                self.model_ft.conv1 = new_features[0]
            input_size = 244
               
        elif model_name == "Resnet50":
            print("[!] Using Resnet50 model")
            self.model_ft = models.resnet50(pretrained=use_pretrained)
            self.set_parameter_requires_grad(self.model_ft, self.feature_extract)
            self.num_of_features = self.model_ft.fc.in_features
            self.model_ft.fc = nn.Linear(self.num_of_features, num_classes)
            if (self.channels == 1):
                new_features = self.model_ft.features[0]
                pretrained_weights = new_features.weight
                layer_conv_1 = nn.Conv2d(1, 64, kernel_size=7, stride=2)
                # For M-channel weight should randomly initialized with Gaussian
                new_features.weight.data.normal_(0, 0.001)
                # For RGB it should be copied from pretrained weights
                #new_features[0].weight.data[:, :3, :, :] = pretrained_weights
            input_size = 244
        
        elif model_name == "Densenet169":
            print("[!] Using Densenet169 model")
            self.model_ft = models.densenet169(pretrained=use_pretrained)
            self.set_parameter_requires_grad(self.model_ft, self.feature_extract)
            #self.model_ft.features = nn.Sequential(*list(self.model_ft.children())[:-1])
            self.model_ft.classifier = (nn.Linear(1664, num_classes))
            if (self.channels == 1):
                new_features = nn.Sequential(*list(self.model_ft.children())[:-1])
                #pretrained_weights = new_features[0].weight
                new_features[0].conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding = 3)
                # For M-channel weight should randomly initialized with Gaussian
                new_features[0].conv0.weight.data.normal_(0, 0.001)
                # For RGB it should be copied from pretrained weights
                #new_features[0].weight.data[:, :3, :, :] = pretrained_weights
                self.model_ft.features = new_features
            
            input_size = 244
            
        elif model_name == "Densenet121":
            print("[!] Using Densenet121 model")
            self.model_ft = models.densenet121(pretrained=use_pretrained)
            self.set_parameter_requires_grad(self.model_ft, self.feature_extract)
            #self.model_ft.features = nn.Sequential(*list(self.model_ft.children())[:-1])
            self.model_ft.classifier = (nn.Linear(1024, num_classes))
            if (self.channels == 1):
                new_features = nn.Sequential(*list(self.model_ft.children())[:-1])
                #pretrained_weights = new_features[0].weight
                new_features[0].conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding = 3)
                # For M-channel weight should randomly initialized with Gaussian
                new_features[0].conv0.weight.data.normal_(0, 0.001)
                # For RGB it should be copied from pretrained weights
                #new_features[0].weight.data[:, :3, :, :] = pretrained_weights
                self.model_ft.features = new_features
            
            input_size = 244
        
        elif model_name == "Densenet201":
            print("[!] Using Densenet201 model")
            self.model_ft = models.densenet201(pretrained=use_pretrained, drop_rate = self.drop_rate)
            self.set_parameter_requires_grad(self.model_ft, self.feature_extract)
            #self.model_ft.features = nn.Sequential(*list(self.model_ft.children())[:-1])
            self.model_ft.classifier = (nn.Linear(1920, num_classes))
            if (self.channels == 1):
                new_features = nn.Sequential(*list(self.model_ft.children())[:-1])
                #pretrained_weights = new_features[0].weight
                new_features[0].conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding = 3)
                # For M-channel weight should randomly initialized with Gaussian
                new_features[0].conv0.weight.data.normal_(0, 0.001)
                # For RGB it should be copied from pretrained weights
                #new_features[0].weight.data[:, :3, :, :] = pretrained_weights
                self.model_ft.features = new_features
            
            input_size = 244
            
        elif model_name == "Densenet161":
            print("[!] Using Densenet161 model")
            self.model_ft = models.densenet161(pretrained=use_pretrained)
            self.set_parameter_requires_grad(self.model_ft, self.feature_extract)
            #self.model_ft.features = nn.Sequential(*list(self.model_ft.children())[:-1])
            self.model_ft.classifier = (nn.Linear(2208, num_classes))
            if (self.channels == 1):
                new_features = nn.Sequential(*list(self.model_ft.children())[:-1])
                #pretrained_weights = new_features[0].weight
                new_features[0].conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding = 3)
                # For M-channel weight should randomly initialized with Gaussian
                new_features[0].conv0.weight.data.normal_(0, 0.001)
                # For RGB it should be copied from pretrained weights
                #new_features[0].weight.data[:, :3, :, :] = pretrained_weights
                self.model_ft.features = new_features
            
            input_size = 244
            
        else:
            print("[x] Invalid model name, exiting!")
            sys.exit()
                    
    def get_model(self):
        return self.model_ft
    
    def set_num_of_layer(self, num_of_layers):
        self.num_of_layers = num_of_layers
    
    def get_num_of_layers(self):
        return self.num_of_layers
    
    def set_parameter_requires_grad(self, model, feature_extracting):  
        if(feature_extracting):
            child_counter = 0
            for child in model.children():
                if child_counter < self.num_of_layers:
                    print("child ",child_counter," was frozen")
                    for param in child.parameters():
                        param.requires_grad = False
                elif child_counter == self.num_of_layers:
                    children_of_child_counter = 0
                    for children_of_child in child.children():
                        if children_of_child_counter < 1:
                            for param in children_of_child.parameters():
                                param.requires_grad = False
                            print('child ', children_of_child_counter, 'of child',child_counter,' was frozen')
                        else:
                            print('child ', children_of_child_counter, 'of child',child_counter,' was not frozen')
                        children_of_child_counter += 1
                else:
                    print("child ",child_counter," was not frozen")
                child_counter += 1
    
    def get_criterion(self, loss_function):
        if loss_function == 'center_loss':
            return CenterLoss(num_classes=3, feat_dim=3, use_gpu=True)
        elif loss_function == 'softmax':
            return nn.Softmax()
        elif loss_function == 'cross_entropy':
            return nn.CrossEntropyLoss()
    
    def get_optimization(self, model, lr, momentum):
            return optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    def get_scheduler(self, optimizer, step_size, gamma):
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    def save_model(self, model, name_model):
        torch.save(model, name_model)
        
    def load_model(self, name_model, localization):
        if(localization == 'cpu'):
            return torch.load(name_model, map_location=lambda storage, loc: storage)
        else:
            return torch.load(name_model)
        
    def get_device(self):
        return self.device
    
    def update_correct_class(self, l1,l2):
        if ((l1 == l2).all()):
            return l1
        if(len(l2)>len(l1)):
            l1_ = l2
            l2_ = l1
        l1_ = set(l1)
        l2_ = set(l2)
        new_list = l1 + list(l2_ - l1_)
        return sorted(new_list) 
    
    def earlier_stop(self, loss):
        #print(self.best_loss, loss)
        if(self.best_loss < loss):
            self.cont_to_stop += 1
        else:
            self.best_loss = loss
            self.cont_to_stop = 0   
            
        if(self.cont_to_stop == 3):
            self.log.log('Loss is not falling down! Stopping this training...', 'l')
            return True
        return False
            
    
    def train_model(self, model, dataloaders, params, dataset_sizes, data, args):
        since = time.time()
        isKfoldMethod = False
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        if args.plot:
            all_features, all_labels = [], []
        
        #Get parameters of training
        lr = params['lr']
        momentum = params['momentum']
        num_epochs = params['num_epochs']
        step_size = params['step_size']
        gamma = params['gamma']
        set_criterion = params['set_criterion']
        net_name = params['net_name']
        drop_rate = params['drop_rate']
        loss_function = params['loss_function']

        #Setting parameters of training
        if(loss_function == 'cross_entropy' or loss_function=='softmax'):
            criterion = self.get_criterion(loss_function)
            optimizer = self.get_optimization(model, lr, momentum)
            scheduler = self.get_scheduler(optimizer, step_size, gamma)
        elif(loss_function == 'center_loss'):
            center_loss = CenterLoss(num_classes=3, feat_dim=3, use_gpu=True)
            criterion = self.get_criterion('cross_entropy')
            params = list(model.parameters()) + list(center_loss.parameters())
            optimizer = torch.optim.SGD(params, lr=lr) # here lr is the overall learning rate
            scheduler = self.get_scheduler(optimizer, step_size, gamma)
            
        #Using more than one GPU
        ######################################
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            #dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)

        model = model.to(self.get_device())
        #####################################
        data.open_file_data(args.save_dir, net_name, lr, drop_rate)
        for epoch in range(num_epochs):
            #os.system('cls' if os.name == 'nt' else 'clear')
            c_print = ''
            #self.log.log('Epoch {}/{}'.format(epoch, num_epochs - 1), 'l')
            
            last_phase = ''
            # Each epoch has a training and validation phase
            for phase in self.folder_names[:2]:
                if phase ==  self.folder_names[0]:
                    scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for i, sample in enumerate(dataloaders[phase]):                                                                                                                                                                                                                                                                                                                                                             
                    (inputs, labels),(filename,_) = sample
                    
                    inputs = inputs.to(self.get_device())
                    labels = labels.to(self.get_device())
                    #print(inputs.shape)

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    alpha = 0.003
                    lr_cent = 0.5

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase ==  self.folder_names[0]):
                        outputs = model(inputs)

                        _, preds = torch.max(outputs, 1)
                        if(loss_function == 'cross_entropy' or loss_function=='softmax'):
                            loss = criterion(outputs, labels) 
                        elif(loss_function == 'center_loss'):
                            loss = center_loss(outputs, labels)*alpha + criterion(outputs, labels)  
                            optimizer.zero_grad()

                        # backward + optimize only if in training phase
                        if phase ==  self.folder_names[0]:
                            loss.backward()
                            if(loss_function == 'center_loss'):
                                for param in center_loss.parameters():
                                    param.grad.data *= (lr_cent/(alpha*lr))
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                    if args.plot:
                        all_features.append(features)
                        all_labels.append(labels.data.cpu().numpy())
                            
                if args.plot:
                    all_features = np.concatenate(all_features, 0)
                    all_labels = np.concatenate(all_labels, 0)
                    plot_features(all_features, all_labels, num_classes, epoch, prefix='train')

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                
                #if phase == 'train':
                    #loss = {'Acc':epoch_acc, 'Loss':epoch_loss}
                    #vis.plot_combine('Combine Plot',loss)
                
                if phase == self.folder_names[0]:
                    c_print = 'Epoch {}/{} : train_loss = {:.4f}, train_acc = {:.4f}'.format(epoch, num_epochs, epoch_loss, epoch_acc)
                else:
                    c_print = c_print + ', val_loss = {:.4f}, val_acc = {:.4f}'.format(epoch_loss, epoch_acc)
                    self.log.log(c_print, 'v')
                content = '{} {:.4f} {:.4f}'.format(epoch, epoch_loss, epoch_acc)
                close = False
                if(epoch == num_epochs - 1):
                    close = True
                data.save_data_training(phase, content, close)
                

                # deep copy the model
                if phase ==  self.folder_names[1] and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                
                last_phase = phase

            if(last_phase == self.folder_names[1]):
                if (self.earlier_stop(epoch_loss)):
                        break
           
            #if keyboard.is_pressed('q'):#if key 'q' is pressed 
             #   print('Stopping this training...')
             #   break#finishing the loop
            #else:
            #    pass

            #print()
        print()
        time_elapsed = time.time() - since
        self.log.log('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60), 'v')
        self.log.log('Best val Acc: {:4f}'.format(best_acc), 'v')

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model
    
    def plot_features(features, labels, num_classes, epoch, prefix):
        """Plot features on 2D plane.
        Args:
            features: (num_instances, num_features).
            labels: (num_instances). 
        """
        colors = ['C0', 'C1', 'C2']
        for label_idx in range(num_classes):
            plt.scatter(
                features[labels==label_idx, 0],
                features[labels==label_idx, 1],
                c=colors[label_idx],
                s=1,
            )
        plt.legend(['0', '1', '2'], loc='upper right')
        dirname = osp.join(args.save_dir, prefix)
        if not osp.exists(dirname):
            os.mkdir(dirname)
        save_name = osp.join(dirname, 'epoch_' + str(epoch+1) + '.png')
        plt.savefig(save_name, bbox_inches='tight')
        plt.close()
    
    def test_model(self, model, dataloaders, folder_name, data):
        import csv
        #logwriter = csv.DictWriter(logfile, fieldnames=['epoch', 'loss', 'val_loss', 'val_acc'])
        #logwriter.writeheader()
        
        was_training = model.training
        model.eval()
        correct = torch.zeros([1, 50], dtype=torch.int32, device = self.device)
        incorrect = torch.zeros([1, 50], dtype=torch.int32, device = self.device)
        results = torch.zeros([50, 50], dtype=torch.int32, device = self.device)
        image_incorrect = [{}]
        
        cont_correct = 0
        cont_incorrect = 0
        correct_class = 0

        with torch.no_grad():      
            for i, sample in enumerate(dataloaders[folder_name]):
                (inputs, labels),(filename,_) = sample
                
                #inputs = inputs.to(self.device)
                #labels = labels.to(self.device)
                
                class_names = data.get_all_image_datasets()[folder_name].classes
                
                labels = [int(class_names[l.item()]) for l in labels]
                correct_class = np.array(list(set(np.array(labels))))

                outputs = model(inputs)
                #print(outputs)
                _, preds = torch.max(outputs, 1)
                
                preds = [int(class_names[l.item()]) for l in preds]
                #print(preds)
                self.log.log("F1 SOCRE:", 'l')
                self.log.log("Macro: {}".format(f1_score(labels, preds, average='macro')), 'v')
                self.log.log("Micro: {}".format(f1_score(labels, preds, average='micro')), 'v')
                self.log.log("Weighted: {}".format(f1_score(labels, preds, average='weighted')), 'v')
                self.log.log("For all analyzed classes: {}".format(f1_score(labels, preds, average=None)), 'v')
                
                                  
                for k in range(len(labels)):
                    
                    if(preds[k] == labels[k]):
                        results[preds[k],preds[k]] +=1
                        correct[0,preds[k]] += 1
                        cont_correct += 1
                    else:
                        results[preds[k],labels[k]] +=1
                        incorrect[0,preds[k]] += 1
                        image_incorrect.append({'class' : preds[k],
                                                'correct_class': labels[k], 
                                                'image': inputs.data[k],
                                                'filename' : filename[k]})
                        cont_incorrect += 1
                        
                if(i>0):
                    correct_class = self.update_correct_class(correct_class_old, correct_class)
                correct_class_old = correct_class

        return results, cont_correct, cont_incorrect, image_incorrect, correct_class