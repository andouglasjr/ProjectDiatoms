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
import utils
from sklearn.metrics import f1_score
from CenterLoss import CenterLoss
import csv
from matplotlib import pyplot as plt

class ModelClass():
    
    def __init__(self, model_name="", channels = 3, num_classes = 50, feature_extract=False, num_of_layers=0, use_pretrained=False, folder_names = None, device = None, log = None):
        self.model_name = model_name
        self.num_classes = num_classes
        self.feature_extract = feature_extract
        self.use_pretrained = use_pretrained
        self.num_of_layers = num_of_layers
        self.channels = channels
        self.model_ft = None
        self.log = log
        input_size = 0
        self.best_loss = 1000
        self.cont_to_stop = 0
        self.num_of_features = 0
        
        
        if(device == None):
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        if (folder_names == None):
            self.folder_names = ['train','val']
        else:
            self.folder_names = folder_names       
        
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
               
        elif model_name == "Resnet101":
            print("[!] Using Resnet101 model")
            self.model_ft = models.resnet101(pretrained=use_pretrained)
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
        else:
            print('Please, what is the loss function?')
            exit(0)
    
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
        if(self.best_loss <= loss):
            self.cont_to_stop += 1
        else:
            self.best_loss = loss
            self.cont_to_stop = 0   
            
        if(self.cont_to_stop == 3):
            self.log.log('Loss is not falling down! Stopping this training...', 'l')
            return True
        return False
    
    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
            
    
    def train_model(self, model, dataloaders, params, data, args):
        since = time.time()
        isKfoldMethod = False
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        if args.plot:
            all_features, all_labels = [], []
        
        logfile = open(args.save_dir+'/'+str(args.network_name[0])+'/logs/log_results_'+str(args.time_training)+'.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['epoch', 'loss', 'val_loss', 'val_acc'])
        logwriter.writeheader()
        
        #Get parameters of training
        lr = params['lr']
        momentum = params['momentum']
        num_epochs = params['num_epochs']
        step_size = params['step_size']
        gamma = params['gamma']
        set_criterion = params['set_criterion']
        net_name = params['net_name']
        loss_function = params['loss_function']
        lr_center_loss = params['lr_center_loss']
        
        lr_batch = []
        loss_batch = []

        #Setting parameters of training
        if(loss_function == 'cross_entropy' or loss_function=='softmax'):
            criterion = self.get_criterion(loss_function)
            optimizer = self.get_optimization(model, lr, momentum)
            scheduler = self.get_scheduler(optimizer, step_size, gamma)
        elif(loss_function == 'center_loss'):
            center_loss = CenterLoss(num_classes=self.num_classes, feat_dim=3, use_gpu=True)
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
        import progressbar
        
        data.open_file_data(args.save_dir, net_name, lr, args)
        for epoch in range(num_epochs):
            folder_epoch = args.save_dir+'/'+net_name+'/lr_'+str(lr)+'_'+str(args.time_training)+'/epochs/epoch_'+str(epoch)+'.pt'
            since_epoch = time.time()
            #os.system('cls' if os.name == 'nt' else 'clear')
            c_print = ''
            #self.log.log('Epoch {}/{}'.format(epoch, num_epochs - 1), 'l')
            
            last_phase = ''
            train_loss = 0
            # Each epoch has a training and validation phase

            for phase in ['train', 'val']  :
                if phase ==  self.folder_names[0]:
                    scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode
                
                
                bar = progressbar.ProgressBar(maxval=len(dataloaders[phase]), \
                        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
                bar.start()
                
                running_corrects = 0
                training_loss = 0.0
                running_loss = 0.0
                print_batch = 1
                # Iterate over data.
                for i, sample in enumerate(dataloaders[phase]):
                    bar.update(i+1)
                    #print(sample)
                    inputs, labels, filename = sample
                    inputs = inputs.to(self.get_device())
                    labels = labels.to(self.get_device())
                    inputs = inputs.repeat(1,3,1,1)
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    alpha = 0.5
                    lr_cent = lr_center_loss

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        if(loss_function == 'cross_entropy' or loss_function=='softmax'):
                            loss = criterion(outputs, labels) 
                        elif(loss_function == 'center_loss'):
                            loss = center_loss(outputs, labels)*alpha + criterion(outputs, labels)  
                            optimizer.zero_grad()

                        # backward + optimize only if in training phase
                        if phase ==  'train':
                            loss.backward()
                            if(loss_function == 'center_loss'):
                                for param in center_loss.parameters():
                                    param.grad.data *= (lr_cent/(alpha*lr))
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                dataset_size = len(data.images_dataset)
                if phase=='train':
                    dataset_size = dataset_size*0.8
                else:
                    dataset_size = dataset_size*0.2
                bar.finish()
                
                epoch_loss = running_loss / dataset_size
                epoch_acc = running_corrects.double() / dataset_size
                                    
                content = '{} {:.4f} {:.4f}'.format(epoch, epoch_loss, epoch_acc)
                close = False
                if(epoch == num_epochs - 1):
                    close = True
                data.save_data_training(phase, content, close)
                
                if phase == 'train':
                    c_print = 'Epoch {}/{} : train_loss = {:.4f}, train_acc = {:.4f}'.format(epoch, num_epochs, epoch_loss, epoch_acc)
                    train_loss, train_acc = epoch_loss, epoch_acc
                else:
                    time_elapsed_epoch = time.time() - since_epoch
                    c_print = c_print + ', val_loss = {:.4f}, val_acc = {:.4f}, Time: {:.0f}m {:.0f}s'.format(epoch_loss, epoch_acc, time_elapsed_epoch // 60, time_elapsed_epoch % 60)
                    self.log.log(c_print, 'v')
                    logwriter.writerow(dict(epoch=epoch, loss=train_loss,
                                    val_loss=epoch_loss, val_acc=epoch_acc.item()))
  
                # deep copy the model
                if phase ==  'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                
                last_phase = phase
                
                self.save_model(model, folder_epoch)

            
            if(last_phase == self.folder_names[1]):
                if (self.earlier_stop(epoch_loss)):
                        break

        logfile.close()
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
    
    def test_model(model, dataloaders, folder_name, data, device, log, args):
        import csv        
        was_training = model.training
        model.eval()
        correct = torch.zeros([1, 50], dtype=torch.int32, device = device)
        incorrect = torch.zeros([1, 50], dtype=torch.int32, device = device)
        results = torch.zeros([50, 50], dtype=torch.int32, device = device)
        image_incorrect = [{}]
        older_model = args.older_model
        
        cont_correct = 0
        cont_incorrect = 0
        correct_class = 0.
        
        class_names = data.get_image_datasets().classes

        vector_transform = [1 ,10,11,12,13,
                         14,15,16,17,18,
                         19,2 ,20,21,22,
                         23,24,25,26,27,
                         28,29,3 ,30,31,
                         32,33,34,35,36,
                         37,38,39,4 ,40,
                         41,42,43,44,45,
                         46,47,48,49,5 ,
                         50, 6, 7, 8, 9]
        
        vector_transform_old = [27, 41, 42]

        with torch.no_grad():      
            for i, sample in enumerate(dataloaders['test']):
                
                if older_model:
                    (inputs, labels),(filename,_) = sample
                    labels = [int(class_names[l.item()]) for l in labels]
                else:
                    inputs, labels, filename = sample
                    inputs = inputs.repeat(1,3,1,1)
                    labels = [(l.item()+1) for l in labels]
                
                #print(labels,filename)
                #inputs = inputs.to(self.device)
                #labels = labels.to(self.device)               
                
                correct_class = np.array(list(set(np.array(labels))))
                print(correct_class)
                
                outputs = model(inputs)
                
                m = nn.Softmax()
                outputs = m(outputs)
                
                _, preds = torch.max(outputs, 1)
                #print(preds)
                
                if older_model:                
                #if(int(args.classes_training) == 3):            
                    preds = [int(vector_transform_old[l.item()]) for l in preds]
                else:
                    #preds = [int(vector_transform[l.item()]) for l in preds]
                    preds = [(p.item()+1) for p in preds]
                #preds = [int(class_names[l.item()]) + 1 for l in preds]
                #else:
                #preds = [(p.item()+1) for p in preds]
                #print(preds)
                import pandas as pd
                df = pd.DataFrame({'Labels' : labels, 'Predictions' : preds, 'Filename': filename})
                #print(df)
                writer = pd.ExcelWriter('report.xlsx')
                df.to_excel(writer, 'Sheet1')
                writer.save()
                    
                log.log("F1 SOCRE:", 'l')
                log.log("Macro: {}".format(f1_score(labels, preds, average='macro')), 'v')
                log.log("Micro: {}".format(f1_score(labels, preds, average='micro')), 'v')
                log.log("Weighted: {}".format(f1_score(labels, preds, average='weighted')), 'v')
                log.log("For all analyzed classes: {}".format(f1_score(labels, preds, average=None)), 'v')
                
                                  
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
                        #print(preds[k], labels[k], filename[k])
                        cont_incorrect += 1
                        
                if(i>0):
                    correct_class = self.update_correct_class(correct_class_old, correct_class)
                correct_class_old = correct_class

        return results, cont_correct, cont_incorrect, image_incorrect, correct_class