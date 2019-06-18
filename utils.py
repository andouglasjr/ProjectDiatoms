import csv
import numpy as np
from matplotlib import pyplot as plt
import torch
from torchvision import transforms
import random
import os
from visdom import Visdom

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
        self.scatter = {}
        
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
            
                
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')
    
    def scatterplot(self, var_name, x, y):
        if var_name not in self.plots:
            self.scatter[var_name] = self.viz.scatter(X=[np.array([x,x]),np.array([y,y])], env=self.env)
        else:
            self.viz.scatter(X=[np.array([x,x]),np.array([y,y])], win=self.scatter[var_name], update = 'append')
      
    
def plot_log(filename, show=True):
    # load data
    keys = []
    values = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if keys == []:
                for key, value in row.items():
                    keys.append(key)
                    values.append(float(value))
                continue

            for _, value in row.items():
                values.append(float(value))

        values = np.reshape(values, newshape=(-1, len(keys)))

    fig = plt.figure()
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(121)

    epoch_axis = 0
    for i, key in enumerate(keys):
        if key == 'epoch':
            epoch_axis = i
            values[:, epoch_axis] += 1
            break
    for i, key in enumerate(keys):
        if key.find('loss') >= 0:  # loss
            print(values[:, i])
            if (key == 'loss'):
            	label = 'Training Loss' 
            else: 
            	label = 'Validation Loss'
            plt.plot(values[:, epoch_axis], values[:, i], label=label, linewidth=2.0)
    plt.legend()
    plt.grid()
    plt.title('Loss Function')

    fig.add_subplot(111)
    for i, key in enumerate(keys):
        if key.find('acc') >= 0:  # acc
        	if (key == 'train_acc'):
        		label = 'Training Acc'
        	else:
        		label = 'Validation Acc'
        	plt.plot(values[:, epoch_axis], values[:, i], label=label, linewidth=2.0)
    plt.legend()
    plt.grid()
    plt.title('Accuracy')

    # fig.savefig('result/log.png')
    if show:
        plt.show()

def plot_log_two_models(filename, show=True):
    # load data
    keys = []
    values = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if keys == []:
                for key, value in row.items():
                    keys.append(key)
                    values.append(float(value))
                continue

            for _, value in row.items():
                values.append(float(value))

        values = np.reshape(values, newshape=(-1, len(keys)))

    fig = plt.figure()
    fig.subplots_adjust(top=0.5, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    styles_1 = ['o-', 'o--']
    styles_2 = ['v-', 'v--']
    epoch_axis = 0
    for i, key in enumerate(keys):
        if key == 'epoch':
            epoch_axis = i
            values[:, epoch_axis] += 1
            break
    for i, key in enumerate(keys):
        if key.find('loss_model1') >= 0:  # loss
            print(values[:, i])
            if (key == 'loss_model1'):
            	label = 'Training - Resenet 50' 
            else: 
            	label = 'Validation - Resenet 50'
            plt.plot(values[:, epoch_axis], values[:, i], styles_1[0], label=label, linewidth=2.0)
        elif key.find('loss_model2') >= 0:  # loss
            print(values[:, i])
            if (key == 'loss_model2'):
            	label = 'Training - Resenet 101' 
            else: 
            	label = 'Validation - Resenet 101'
            plt.plot(values[:, epoch_axis], values[:, i], styles_2[1], label=label, linewidth=2.0)
    plt.annotate(
        'Best Epoch Resnet50',
        xy=(16,0.0522521154), arrowprops=dict(arrowstyle='->'), xytext=(10, 1.5))
    
    plt.annotate(
        'Best Epoch Resnet101',
        xy=(6,0.0129342183), arrowprops=dict(arrowstyle='->'), xytext=(6, 1))

    plt.legend()
    plt.grid(which='both', linestyle='--')
    
    plt.title('Loss Function')

    fig.add_subplot(212)
    for i, key in enumerate(keys):
        
        if key.find('acc_model1') >= 0:  # acc
        	if (key == 'train_acc_model1'):
        		label = 'Training - Resnet 50'
        	else:
        		label = 'Validation - Resnet 50'
        	plt.plot(values[:, epoch_axis], values[:, i], styles_1[0], label=label, linewidth=2.0)
        elif key.find('acc_model2') >= 0:  # acc
            if (key == 'train_acc_model2'):
                label = 'Training - Resnet 100'
            else:
                label = 'Validation - Resnet 100'
            plt.plot(values[:, epoch_axis], values[:, i],  styles_2[1], label=label, linewidth=2.0)
            
    plt.annotate(
        'Best Epoch Resnet50',
        xy=(16,0.998), arrowprops=dict(arrowstyle='->'), xytext=(10, 0.7))
    
    plt.annotate(
        'Best Epoch Resnet101',
        xy=(6,0.99975), arrowprops=dict(arrowstyle='->'), xytext=(6, 0.8))
    
    plt.legend()
    plt.grid(which='both', linestyle='--')
    plt.title('Accuracy')

    # fig.savefig('result/log.png')
    if show:
        plt.show()


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

def get_learning_rate(args, network_name):
    if args.exponential_range is not None:
        return 10**random.uniform(args.exponential_range[0], args.exponential_range[1])
    if args.new_lr:
        return args.lr
    else:
        if(network_name == "Resnet50" or network_name == "Resnet101" or "Resnet18"):
            return 3.118464108103618e-05
        elif(network_name == "Densenet201"):
            return 8.832537199285954e-04
        elif(network_name == "Densenet169"):
            return 2.6909353460670058e-05

def create_folders_to_results(args):
    ################################################################################################################
    #Folder names to save the log and model
    ################################################################################################################
    localtime = args.time_training
    lr = args.lr
    folder_save_results_epoch = args.save_dir+'/'+args.network_name+'/lr_'+str(lr)+'_'+str(localtime)+'/epochs'

    if not os.path.exists(folder_save_results_epoch):
        os.makedirs(folder_save_results_epoch)

    folder_best_result = args.save_dir + '/'+args.network_name+'/lr_'+ str(lr)+'_'+str(localtime)+'/best_result.pt'
    return folder_save_results_epoch, folder_best_result



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def plot_bar_chart(model, inputs, title):
    mean, std = [0.5017, 0.5017, 0.5017],[0.1057, 0.1057, 0.1057]
    transform_compose = transforms.Compose([transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean, std)
                                          ])
    inputs = transform_compose(inputs)
    inputs = torch.unsqueeze(inputs, 0)
    inputs = inputs.repeat(1,3,1,1)
    classes = [x for x in range(1,51)]
    
    m = torch.nn.Softmax()
    outputs = model[0](inputs)
    out_softmax_1 = m(outputs)

    outputs = model[1](inputs)
    out_softmax_2 = m(outputs)
    out = torch.add(out_softmax_1, out_softmax_2)

    #outputs = model[2](inputs)
    #out_softmax_3 = m(outputs)
    #out = torch.add(out, out_softmax_3)
    
    #outputs = model(inputs)
    
    #m = torch.nn.Softmax()
    o = m(out)
    o = o.cpu()
    o = o.detach().numpy()

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(len(classes))
    bar_width = 0.3
    opacity = 0.8
    
    #################
    rects1 = plt.bar(index, out_softmax_1.cpu().detach().numpy()[0], bar_width,
                    alpha=opacity,
                    color='b',
                    label='Resnet50')

    #################
    rects2 = plt.bar(index+bar_width, out_softmax_2.cpu().detach().numpy()[0], bar_width,
                    alpha=opacity,
                    color='r',
                    label='Resnet101_90')
    
    rects2 = plt.bar(index+bar_width+bar_width, out.cpu().detach().numpy()[0], bar_width,
                    alpha=opacity,
                    color='g',
                    label='Sum')
    
    #################
    #rects2 = plt.bar(index+bar_width+bar_width, out_softmax_3.cpu().detach().numpy()[0], bar_width,
    #                alpha=opacity,
    #                color='g',
    #                label='Resnet101_88')
    
    plt.xlabel('Scores')
    plt.ylabel('%')
    plt.title(title)
    plt.xticks(index + bar_width, classes)
    plt.legend()

    plt.tight_layout()
    #plt.show()
    
    #y_pos = np.arange(len(classes))
    #plt.bar(y_pos, o[0], align='center', alpha=0.5)
    #plt.xticks(y_pos, classes)
    #plt.ylabel('%')
    #plt.title('Output Softmax')
    #plt.show()

if __name__=="__main__":
    plot_log("results/S/logs/log_results_Thu Jun 13 17:36:14 2019.csv")
    from DataUtils import DataUtils
    import utils
    import torch
    from PIL import Image
    
    #images_name = ["../data/Dataset_5/Diatom50NEW_generated/test_diatoms_3_class/27/TestIm27_10.png",
    #    "../data/Dataset_5/Diatom50NEW_generated/test_diatoms_3_class/41/Phase41_new_10.png",
    #    "../data/Dataset_5/Diatom50NEW_generated/test_diatoms_3_class/41/TestIm41_11.png", 
    #              "../data/Dataset_5/Diatom50NEW_generated/test_diatoms_3_class/41/TestIm41_11.png"]

    #image_name = "../data/Dataset_5/Diatom50NEW_generated/test_diatoms_3_class/42/TestIm42_8.png"
    #image = Image.open(image_name)
    
    #model_names = ["results/Resnet50/lr_0.0003118464108103618_Mon Feb 25 20:01:50 2019/epochs/epoch_15.pt", 
    #                  "results/Resnet101/lr_0.0003118464108103618_Fri Feb 22 14:07:07 2019/epochs/epoch_5.pt",
    #                  "results/Resnet101/lr_0.0003118464108103618_Thu Feb 21 11:01:00 2019/epochs/epoch_3.pt"]
    #model = []
    #model.append(torch.load(model_names[0]))
    #model.append(torch.load(model_names[1]))
    #model.append(torch.load(model_names[2]))
    
    #for image_name in images_name:
    #    image = Image.open(image_name)
    #    utils.plot_bar_chart(inputs=image, model=model, title = image_name)
    
    #plt.show()
    
    
