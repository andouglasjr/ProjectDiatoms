import torch
import progressbar
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import csv
import time
import copy
import utils
from ModelUtils import ModelUtils
from DataLogger import DataLogger
from Ensemble import Ensemble

class TrainingClass():
    
    def __init__(self, model, data, args):
        self.model = model
        self.dataloader = data.load_data() 
        self.data = data
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.folder_names = ['train','val']
        self.log = DataLogger.getInstance(args)
        
        #Log variables
        self.logfile = open(args.save_dir+'/'+str(args.network_name[0])+'/logs/log_results_'+str(args.time_training)+'.csv', 'w')
        self.logwriter = csv.DictWriter(self.logfile, fieldnames=['x', 'train_loss', 'val_loss', 'train_acc','val_acc'])
        self.x_log = []
        self.train_loss_log = []
        self.val_loss_log = []
        self.train_acc_log = []
        self.val_acc_log = []
        self.logwriter.writeheader()
        
        #Set parameters
        self.lr = self.args.lr
        self.momentum = float(self.args.momentum)
        self.num_epochs = self.args.epochs
        self.step_size = int(self.args.step_size)
        self.gamma = float(self.args.gamma)
        self.net_name = self.args.network_name
        self.loss_function = self.args.loss_function
    
    def fit_ensemble(self, models):
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import VotingClassifier

        log_clf = LogisticRegression()
        rnd_clf = RandomForestClassifier()
        svm_clf = SVC()

        voting_clf = VotingClassifier(
            estimators = [('lr', log_clf), ('rf', rnd_clf), ('svc',svm_clf)],
            voting = 'hard')
        
        model_1 = models[0].to(self.device)
        model_2 = models[1].to(self.device)
        
        for phase in ['train', 'val'] :               
            for i, sample in enumerate(self.dataloader[phase]):
                inputs, labels, filename, shape = sample
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                inputs = inputs.repeat(1,3,1,1)
                
                
                x_1 = model_1(inputs)
                x_2 = model_2(inputs)
                x = torch.cat((x_1,x_2), dim=1)
                
                voting_clf.fit(x.cpu().detach().numpy(), labels)
                
        return voting_clf
        
    
    def train(self):
        
        #Using more than one GPU
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            #dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = nn.DataParallel(self.model)

        self.model = self.model.to(self.device)
        
        since = time.time()
        isKfoldMethod = False
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        best_epoch_pos = 0
        
        global plotter
        plotter = utils.VisdomLinePlotter(env_name='Plot')
        
        lr_batch = []
        loss_batch = []
        plot_x_train = 0

        #Setting parameters of training
        criterion = ModelUtils.get_criterion(self.loss_function)
        optimizer = ModelUtils.get_optimization(self.model, self.lr, self.momentum)
        scheduler = ModelUtils.get_scheduler(optimizer, self.step_size, self.gamma)        
        
        self.data.open_file_data(self.args.save_dir, self.net_name, self.lr, self.args)
        
        for epoch in range(self.num_epochs):
            folder_epoch = self.args.save_dir+'/'+self.net_name+'/lr_'+str(self.lr)+'_'+str(self.args.time_training)+'/epochs/epoch_'+str(epoch)+'.pt'
            since_epoch = time.time()
            c_print = ''
            last_phase = ''
            train_loss = 0
            
            # Each epoch has a training and validation phase
            for phase in ['train', 'val'] :
                if phase ==  self.folder_names[0]:
                    scheduler.step()
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode
                
                bar = progressbar.ProgressBar(maxval=len(self.dataloader[phase]), \
                        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
                bar.start()
                
                running_corrects = 0
                training_loss = 0.0
                running_loss = 0.0
                print_batch = 1
                dataset_size = 0
                # Iterate over data.
                n_batchs = len(self.data.images_dataset)/self.args.batch_size
                for i, sample in enumerate(self.dataloader[phase]):
                    bar.update(i+1)
                    #print(sample)
                    inputs, labels, filename, shape = sample
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    inputs = inputs.repeat(1,3,1,1)
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs= self.model(inputs)
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
                    
                    dataset_size += inputs.size(0)
                    plot_x_train += 1/n_batchs
                    
                    _loss = running_loss/dataset_size
                    _acc = running_corrects.double()/dataset_size
                    
                    if phase == 'train':
                        plotter.plot('Loss', 'Train', 'Loss Training', plot_x_train, _loss)
                        plotter.plot('Accuracy', 'Train', 'Accuracy Training', plot_x_train, _acc)
                        self.x_log.append(plot_x_train)
                        self.train_loss_log.append(_loss)
                        self.val_loss_log.append(0)
                        self.train_acc_log.append(_acc)
                        self.val_acc_log.append(0)
                        #logwriter.writerow(dict(x=plot_x_train, train_loss=_loss, val_loss=0, train_acc=_acc.item(), val_acc = 0))
                    else:
                        plotter.plot('Loss', 'Validation', 'Loss Training', plot_x_train, _loss)
                        plotter.plot('Accuracy', 'Validation', 'Accuracy Training', plot_x_train, _acc)
                        #logwriter.writerow(dict(x=plot_x_train, train_loss=0, val_loss=_loss, train_acc=0, val_acc = _acc.item()))
                        self.x_log.append(plot_x_train)
                        self.train_loss_log.append(0)
                        self.val_loss_log.append(_loss)
                        self.train_acc_log.append(0)
                        self.val_acc_log.append(_acc)
                        
                bar.finish()
                
                epoch_loss = running_loss / dataset_size
                epoch_acc = running_corrects.double() / dataset_size                                    
                content = '{} {:.4f} {:.4f}'.format(epoch, epoch_loss, epoch_acc)
                close = False
                
                if(epoch == self.num_epochs - 1):
                    close = True
                self.data.save_data_training(phase, content, close)
                
                if phase == 'train':
                    c_print = 'Epoch {}/{} : train_loss = {:.4f}, train_acc = {:.4f}'.format(epoch, self.num_epochs, epoch_loss, epoch_acc)
                    train_loss, train_acc = epoch_loss, epoch_acc
                else:
                    time_elapsed_epoch = time.time() - since_epoch
                    c_print = c_print + ', val_loss = {:.4f}, val_acc = {:.4f}, Time: {:.0f}m {:.0f}s'.format(epoch_loss, epoch_acc, time_elapsed_epoch // 60, time_elapsed_epoch % 60)
                    self.log.log(c_print, 'v')
                    #logwriter.writerow(dict(x=plot_x_train, train_loss=0, val_loss=_loss, train_acc=0, val_acc = _acc.item()))
                    self.logwriter.writerow(dict(x=self.x_log, train_loss=self.train_loss_log, val_loss=self.val_loss_log, train_acc=self.train_acc_log, val_acc = self.val_acc_log))
                    
  
                # deep copy the model
                if phase ==  'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_epoch_pos = epoch
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                
                last_phase = phase
                
                ModelUtils.save_model(self.model, folder_epoch)

            
            if(last_phase == self.folder_names[1]):
                if (ModelUtils.earlier_stop(epoch_loss)):
                        break

        self.logfile.close()
        print()
        time_elapsed = time.time() - since
        self.log.log('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60), 'v')
        self.log.log('Best val Acc: {:4f} in Epoch: {}'.format(best_acc, best_epoch_pos), 'v')

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model