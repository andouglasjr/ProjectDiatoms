import argparse
import os
import time

class ArgumentsParser():
    def __init__(self):
        # setting the hyper parameters
        self.localtime = time.asctime(time.localtime(time.time()))
        parser = argparse.ArgumentParser(description="Diatoms Research CNR-ISASI")
        parser.add_argument('--network_name',  
                            help="Insert the network name for trainning, e.g. --network_name Resnet18")

        parser.add_argument('--new_lr', 
                            action='store_true',
                           help="If not setted will be used the lr already found!")

        parser.add_argument('--range', 
                            default=1, 
                            type=int,
                           help="How much times the network will be trainned with diferent learning rates!")

        parser.add_argument('--exponential_range', 
                            nargs='*', 
                            default=None, 
                            type=float,
                           help="What is the exponential range to find the best lr, e.g. --exponential_range -3 -5")

        parser.add_argument('--epochs', 
                            default=8, 
                            type=int,
                           help="Number of epochs")

        parser.add_argument('--batch_size', 
                            default=256, 
                            type=int,
                           help="Size of the batch")

        parser.add_argument('--lr', 
                            default=0.0001, 
                            type=float,
                            help="Initial learning rate")

        parser.add_argument('--older_model', 
                            action='store_true',
                           help="The model was obtained before the changes in the augmentation phase?")

        parser.add_argument('--new_aug', 
                            action='store_true',
                           help="Add the new augmentation!")

        parser.add_argument('--loss_function', 
                            default = 'cross_entropy',
                           help="Which will be loss function used? Cross Validation (default), Center Loss or both")

        parser.add_argument('--data_dir', 
                            default='../data/Dataset_5/Diatom50NEW_generated',
                            help="Directory of data. If no data, use \'--download\' flag to download it")

        parser.add_argument('--save_dir', 
                            default='results',
                           help="Directory to save the results!")

        parser.add_argument('-t', 
                            '--testing', 
                            action='store_true',
                            help="Test the trained model on testing dataset")

        parser.add_argument('-w', 
                            '--weights', 
                            default=None,
                            help="The path of the saved weights. Should be specified when testing")

        parser.add_argument('--plot', 
                            action='store_true', 
                            help="whether to plot features for every epoch")

        parser.add_argument('--images_per_class', 
                            default=21000, 
                            help="how many images will be used per class")

        parser.add_argument('--classes_training', 
                            default=50, 
                            help="how many classes there are in the train")

        parser.add_argument('--time_training', 
                            default=self.localtime, 
                            help="start time training")

        self.args = parser.parse_args()
        self.print_args()
    
    def get_args(self):
        return self.args
    
    def print_args(self):
        ts = os.get_terminal_size()
        print(''.center(ts.columns, '-'))
        print("Setup Training - {}".format(self.localtime).center(ts.columns, ' '))
        print(''.center(ts.columns, '-'))
        print("     {0:<10}".format("Network Name:.................." + self.args.network_name))
        print("     {0:<10}".format("Epochs:........................" + str(self.args.epochs)))
        print("     {0:<10}".format("Batch Size:...................." + str(self.args.batch_size)))
        print("     {0:<10}".format("Loss Function:................." + self.args.loss_function))
        print("     {0:<10}".format("Data Directory:................" + self.args.data_dir))
        print("     {0:<10}".format("Images per Class:.............." + str(self.args.images_per_class)))
        print("     {0:<10}".format("Number of Classes:............." + str(self.args.classes_training)))
        print(''.center(ts.columns, '-'))
        print("Activated Flags".format(self.localtime).center(ts.columns, ' '))
        print(''.center(ts.columns, '-').center(ts.columns, ' '))
        content = ""
        if(self.args.new_lr):
            content+="new_lr - "
        if(self.args.older_model):
            content+="older_model - "
        if(self.args.new_aug):
            content+="new_aug - "
        if(self.args.testing):
            content+="testing"
        if content is "":
            print("Nothing".center(ts.columns, ' '))
        else:
            print("{}".format(content).center(ts.columns, ' '))
        print(''.center(ts.columns, '-'))
        