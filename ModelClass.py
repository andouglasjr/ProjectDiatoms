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
from DiatomsNetwork import DiatomsNetwork
import torchvision
from FullyConnectedCapsuled import FullyConnectedCapsuled
import pandas as pd
from DataUtils import DataUtils
from sklearn.metrics import accuracy_score
import utils

class ModelClass():
    
    def __init__(self, model_name="", channels = 3, num_classes = 50, feature_extract=False, num_of_layers=0, use_pretrained=True, folder_names = None, device = None, log = None):
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
            #self.model_ft.fc = FullyConnectedCapsuled(self.num_of_features, num_classes)
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
        
        elif model_name == "DiatomsNetwork":
            print("[!] DiatomsNetwork model")
            self.model_ft = DiatomsNetwork(num_classes)
            
            input_size = 244
                    
        elif model_name == "SqueezeNet":
            print("[!] SqueezeNet model")
            self.model_ft = models.squeezenet1_1(pretrained=use_pretrained)
            
            #for name, params in self.model_ft.named_children():
            #    print(name)
            
            in_ftrs = self.model_ft.classifier[1].in_channels
            #print(in_ftrs)
            out_ftrs = self.model_ft.classifier[1].out_channels
            #print(out_ftrs)
            features = list(self.model_ft.classifier.children())
            features[1] = nn.Conv2d(in_ftrs, num_classes,1,1)
            features[3] = nn.AvgPool2d(13,stride=1)
            
            self.model_ft.classifier = nn.Sequential(*features)
            self.model_ft.num_classes = num_classes
            
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
            
        if(self.cont_to_stop == 2):
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
        best_epoch_pos = 0
        if args.plot:
            all_features, all_labels = [], []
        
        logfile = open(args.save_dir+'/'+str(args.network_name[0])+'/logs/log_results_'+str(args.time_training)+'.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['x', 'train_loss', 'val_loss', 'train_acc','val_acc'])
        x_log = []
        train_loss_log = []
        val_loss_log = []
        train_acc_log = []
        val_acc_log = []
        logwriter.writeheader()
        
        global plotter
        plotter = utils.VisdomLinePlotter(env_name='Plot')
        
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
        plot_x_train = 0

        #Setting parameters of training
        criterion = self.get_criterion(loss_function)
        optimizer = self.get_optimization(model, lr, momentum)
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
            

            for phase in ['train', 'val'] :
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
                dataset_size = 0
                # Iterate over data.
                n_batchs = len(data.images_dataset)/args.batch_size
                for i, sample in enumerate(dataloaders[phase]):
                    bar.update(i+1)
                    #print(sample)
                    inputs, labels, filename, shape = sample
                    inputs = inputs.to(self.get_device())
                    labels = labels.to(self.get_device())
                    inputs = inputs.repeat(1,3,1,1)
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs= model(inputs)
                        #print(outputs.size(), labels.size(), inputs.size())
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels) 
                        optimizer.zero_grad()

                        # backward + optimize only if in training phase
                        if phase ==  'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                    #batch_loss = running_loss / inputs.
                    
                    dataset_size += inputs.size(0)
                    plot_x_train += 1/n_batchs
                    
                    _loss = running_loss/dataset_size
                    _acc = running_corrects.double()/dataset_size
                    
                    if phase == 'train':
                        plotter.plot('Loss', 'Train', 'Loss Training', plot_x_train, _loss)
                        plotter.plot('Accuracy', 'Train', 'Accuracy Training', plot_x_train, _acc)
                        x_log.append(plot_x_train)
                        train_loss_log.append(_loss)
                        val_loss_log.append(0)
                        train_acc_log.append(_acc)
                        val_acc_log.append(0)
                        #logwriter.writerow(dict(x=plot_x_train, train_loss=_loss, val_loss=0, train_acc=_acc.item(), val_acc = 0))
                    else:
                        plotter.plot('Loss', 'Validation', 'Loss Training', plot_x_train, _loss)
                        plotter.plot('Accuracy', 'Validation', 'Accuracy Training', plot_x_train, _acc)
                        #logwriter.writerow(dict(x=plot_x_train, train_loss=0, val_loss=_loss, train_acc=0, val_acc = _acc.item()))
                        x_log.append(plot_x_train)
                        train_loss_log.append(0)
                        val_loss_log.append(_loss)
                        train_acc_log.append(0)
                        val_acc_log.append(_acc)
                    #print(running_loss / dataset_size)
                #dataset_size = len(data.images_dataset)
                #if phase=='train':
                #    dataset_size = dataset_size*0.8
                #else:
                #    dataset_size = dataset_size*0.2
                bar.finish()
                
                epoch_loss = running_loss / dataset_size
                epoch_acc = running_corrects.double() / dataset_size
                #plot_x = 0
                #plotter.scatterplot("Loss Training", epoch, epoch_loss)
                                    
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
                    #logwriter.writerow(dict(x=plot_x_train, train_loss=0, val_loss=_loss, train_acc=0, val_acc = _acc.item()))
                    logwriter.writerow(dict(x=x_log, train_loss=train_loss_log, val_loss=val_loss_log, train_acc=train_acc_log, val_acc = val_acc_log))
                    
  
                # deep copy the model
                if phase ==  'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_epoch_pos = epoch
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
        self.log.log('Best val Acc: {:4f} in Epoch: {}'.format(best_acc, best_epoch_pos), 'v')

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
        
        correct = torch.zeros([1, 51], dtype=torch.int32, device = device)
        incorrect = torch.zeros([1, 51], dtype=torch.int32, device = device)
        results = torch.zeros([51, 51], dtype=torch.int32, device = device)
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
        
        
        y_pred = []
        y_test = []
        correct_class = []
        with torch.no_grad():      
            
            for i, sample in enumerate(dataloaders['test']):
                
                if older_model:
                    (inputs, labels),(filename,_) = sample
                    labels = [int(class_names[l.item()]) for l in labels]
                else:
                    inputs, labels, filename, shape = sample
                    inputs = inputs.repeat(1,3,1,1)
                    #labels = [(l.item()+1) for l in labels]
                
                #print(labels)
                #inputs = inputs.to(self.device)
                #labels = labels.to(self.device)               
                correct_class = np.concatenate((correct_class, labels),0)
                correct_class = sorted(np.array(list(set(np.array(correct_class,dtype=np.int16))),dtype=np.int16))
                print(correct_class)
                
                
                outputs = model(inputs)
                
                _, preds = torch.max(outputs, 1)
                print(preds)
                
                if older_model:                
                #if(int(args.classes_training) == 3):            
                    preds = [int(vector_transform_old[l.item()]) for l in preds]
                else:
                    #preds = [int(vector_transform_old[l.item()]) for l in preds]
                    preds = [(p.item()+1) for p in preds]
                    #preds = [int(p.item()) for p in preds]
                    
                #preds = [int(class_names[l.item()]) + 1 for l in preds]
                #else:
                #preds = [(p.item()+1) for p in preds]
                #print(preds)
                
                #df = pd.DataFrame({'Labels' : labels, 'Predictions' : preds, 'Filename': filename})
                #print(df)
                #writer = pd.ExcelWriter('report.xlsx')
                #df.to_excel(writer, 'Sheet1')
                #writer.save()
                    
                
                y_pred = np.concatenate((y_pred,preds),0)
                #print(y_pred)
                y_test = np.concatenate((y_test,labels),0)
                #print(y_test)
                                               
                #if(i>0):
                #    correct_class = self.update_correct_class(correct_class_old, correct_class)
                #correct_class_old = correct_class
                
                
            if not older_model:
                y_test = [l + 1 for l in y_test]
                #y_pred = [p + 1 for p in y_pred]
            
            log.log("F1 SOCRE:", 'l')
            log.log("Macro: {}".format(f1_score(y_test, y_pred, average='macro')), 'v')
            log.log("Micro: {}".format(f1_score(y_test, y_pred, average='micro')), 'v')
            log.log("Weighted: {}".format(f1_score(y_test, y_pred, average='weighted')), 'v')
            log.log("For all analyzed classes: {}".format(f1_score(y_test, y_pred, average=None)), 'v')
            
            class_names = sorted(np.array(list(set(np.array(y_pred)))))
            #data.confusion_matrix_skt(y_test = y_test, y_pred = y_pred, class_names=class_names)
            labels = np.array(y_test, dtype=int)
            preds = np.array(y_pred, dtype = int)
            
            for k in range(len(labels)-1):

                if(preds[k] == labels[k]):
                    results[preds[k],preds[k]] +=1
                    correct[0,preds[k]] += 1
                    cont_correct += 1
                else:
                    results[preds[k],labels[k]] +=1
                    incorrect[0,preds[k]] += 1
                    #image_incorrect.append({'class' : preds[k],
                    #                        'correct_class': labels[k], 
                    #                        'image': inputs.data[k],
                    #                        'filename' : filename[k]})
                    #print(preds[k], labels[k], filename[k])
                    cont_incorrect += 1

            
            

        return results, cont_correct, cont_incorrect, image_incorrect, correct_class
    
    def test_models(model, dataloaders, folder_name, data, device, log, args):
        import csv        
        was_training = model[0].training
        model[0].eval()
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
        y_pred = []
        y_test = []
        correct_class = []
        
        

        with torch.no_grad():      
            for i, sample in enumerate(dataloaders['test']):
                
                if older_model:
                    (inputs, labels),(filename,_) = sample
                    labels = [int(class_names[l.item()]) for l in labels]
                else:
                    inputs, labels, filename, shape = sample
                    inputs = inputs.repeat(1,3,1,1)
                    #labels = [(l.item()+1) for l in labels]
                
                print(labels)
                #inputs = inputs.to(self.device)
                #labels = labels.to(self.device)               
                
                correct_class = np.array(list(set(np.array(labels))))
                print(correct_class)

                m = nn.Softmax()
                outputs = model[0](inputs)
                out1 = m(outputs)
                
                outputs = model[1](inputs)
                out2 = m(outputs)
                out = torch.add(out1, out2)
  
                #print(out, out1, out2, filename)
                #outputs = model[2](inputs)
                #out = torch.add(out, m(outputs))
                
                _, preds = torch.max(out, 1)
                
                
                
                if older_model:                
                #if(int(args.classes_training) == 3):            
                    preds = [int(vector_transform_old[l.item()]) for l in preds]
                else:
                    #preds = [int(vector_transform_old[l.item()]) for l in preds]
                    #preds = [(p.item()+1) for p in preds]
                    preds = [(p.item()) for p in preds]
                #preds = [int(class_names[l.item()]) + 1 for l in preds]
                #else:
                #preds = [(p.item()+1) for p in preds]
                #print(preds)
                y_pred = np.concatenate((y_pred,preds),0)
                #print(y_pred)
                y_test = np.concatenate((y_test,labels),0)
                #print(y_test)
                                               
                #if(i>0):
                #    correct_class = self.update_correct_class(correct_class_old, correct_class)
                #correct_class_old = correct_class
                
                
            if not older_model:
                y_test = [l + 1 for l in y_test]
                y_pred = [p + 1 for p in y_pred]
            
            log.log("F1 SOCRE:", 'l')
            log.log("Macro: {}".format(f1_score(y_test, y_pred, average='macro')), 'v')
            log.log("Micro: {}".format(f1_score(y_test, y_pred, average='micro')), 'v')
            log.log("Weighted: {}".format(f1_score(y_test, y_pred, average='weighted')), 'v')
            log.log("For all analyzed classes: {}".format(f1_score(y_test, y_pred, average=None)), 'v')
            
            class_names = sorted(np.array(list(set(np.array(y_pred)))))
            data.confusion_matrix_skt(y_test = y_test, y_pred = y_pred, class_names=class_names)
            accuracy = accuracy_score(y_test, y_pred)
            print("Accuracy: %.2f%%" % (accuracy * 100.0))
            
            
            
            labels = np.array(y_test, dtype=int)
            preds = np.array(y_pred, dtype = int)
            
            for k in range(len(labels)):

                if(preds[k] == labels[k]):
                    results[preds[k],preds[k]] +=1
                    correct[0,preds[k]] += 1
                    cont_correct += 1
                else:
                    results[preds[k],labels[k]] +=1
                    incorrect[0,preds[k]] += 1
                    #image_incorrect.append({'class' : preds[k],
                    #                        'correct_class': labels[k], 
                    #                        'image': inputs.data[k],
                    #                        'filename' : filename[k]})
                    #print(preds[k], labels[k], filename[k])
                    cont_incorrect += 1


        return results, cont_correct, cont_incorrect, image_incorrect, correct_class
    
    def test_models_votting(model, dataloaders, folder_name, data, device, log, args):
        import csv        
        #was_training = model[0].training
        #model[0].eval()
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
        y_pred = []
        y_test = []
        correct_class = []
        
        

        with torch.no_grad():      
            for i, sample in enumerate(dataloaders['test']):
                
                if older_model:
                    (inputs, labels),(filename,_) = sample
                    labels = [int(class_names[l.item()]) for l in labels]
                else:
                    inputs, labels, filename, shape = sample
                    inputs = inputs.repeat(1,3,1,1)
                    #labels = [(l.item()+1) for l in labels]
                
                print(labels)
                #inputs = inputs.to(self.device)
                #labels = labels.to(self.device)               
                
                correct_class = np.array(list(set(np.array(labels))))
                print(correct_class)
                
                model_1 = model[0]
                model_2 = model[1]

                from Ensemble import Ensemble 
                model_ensemble = Ensemble(model_1, model_2)
                out = model_ensemble(inputs)
                print(out)
  
                #print(out, out1, out2, filename)
                #outputs = model[2](inputs)
                #out = torch.add(out, m(outputs))
                
                _, preds = torch.max(out, 1)
                
                
                
                if older_model:                
                    preds = [int(vector_transform_old[l.item()]) for l in preds]
                else:

                    preds = [(p.item()) for p in preds]

                y_pred = np.concatenate((y_pred,preds),0)
                y_test = np.concatenate((y_test,labels),0)

                
            if not older_model:
                y_test = [l + 1 for l in y_test]
                y_pred = [p + 1 for p in y_pred]
            
            log.log("F1 SOCRE:", 'l')
            log.log("Macro: {}".format(f1_score(y_test, y_pred, average='macro')), 'v')
            log.log("Micro: {}".format(f1_score(y_test, y_pred, average='micro')), 'v')
            log.log("Weighted: {}".format(f1_score(y_test, y_pred, average='weighted')), 'v')
            log.log("For all analyzed classes: {}".format(f1_score(y_test, y_pred, average=None)), 'v')
            
            class_names = sorted(np.array(list(set(np.array(y_pred)))))
            data.confusion_matrix_skt(y_test = y_test, y_pred = y_pred, class_names=class_names)
            accuracy = accuracy_score(y_test, y_pred)
            print("Accuracy: %.2f%%" % (accuracy * 100.0))
    
    def get_features_layer(model, data, device):
        if model == None:
            model = torch.load(model_names[0])

        model = model.module
        feature_layer_model = nn.Sequential(*list(model.children())[:-1])
        #Using more than one GPU
        ######################################
        if torch.cuda.device_count() > 1:
            #print("Let's use", torch.cuda.device_count(), "GPUs!")
            #dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            feature_layer_model = nn.DataParallel(feature_layer_model)

        feature_layer_model = feature_layer_model.to(device)
        #####################################
        #data = torch.from_numpy(data)
        feature_layer_output = feature_layer_model(data)
        feature_layer_output = torch.squeeze(feature_layer_output)
        return feature_layer_output
    
    def test_xgboost_model(model, xgb_model, dataloaders, folder_name, data, device, log, args):
        import csv        
        was_training = model[0].training
        model[0].eval()
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
        
        
        model_1 = model[0]
        model_1 = model_1.to(device)
        model_2 = model[1]
        model_2 = model_2.to(device)
        
        with torch.no_grad():      
            for i, sample in enumerate(dataloaders['test']):
                
                if older_model:
                    (inputs, labels),(filename,_) = sample
                    labels = [int(class_names[l.item()]) for l in labels]
                else:
                    inputs, labels, filename, shape = sample
                    inputs = inputs.repeat(1,3,1,1)
                    #labels = [(l.item()+1) for l in labels]
                
                print(labels)
                #inputs = inputs.to(self.device)
                #labels = labels.to(self.device)               
                
                correct_class = np.array(list(set(np.array(labels))))
                print(correct_class)
                
                feat_out_1 = ModelClass.get_features_layer(model_1, inputs, device)
                feat_out_2 = ModelClass.get_features_layer(model_2, inputs, device)
                feat_out = torch.cat((feat_out_1, feat_out_2), 1)

                y_pred = xgb_model.predict(feat_out)
                predictions = [round(value) for value in y_pred]
                print(predictions)
                # evaluate predictions
                accuracy = accuracy_score(labels, predictions)
                print("Accuracy: %.2f%%" % (accuracy * 100.0))
                
            