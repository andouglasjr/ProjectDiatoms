from torch.autograd import Variable
import torch.nn.functional as F
import torch
from FullyConnectedCapsuled import FullyConnectedCapsuled

class DiatomsNetwork(torch.nn.Module):
    
    #Batch shape for input x is (3, 224, 224)
    
    def __init__(self, num_class = 3):
        super(DiatomsNetwork, self).__init__()
        
        #self.num_class = 6
        #Inputs channels = 3, outputs channels = 18
        self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=5, stride=1, padding=0) #224
        
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0) #110
        
        self.conv2 = torch.nn.Conv2d(18, 36, kernel_size=7, stride=1, padding=0) #112        
        
        #4608 input features, 64 outputs features
        self.fc1 = torch.nn.Linear(18 * 52 * 52, 512)
        
        #self.fcc = FullyConnectedCapsuled(num_class)
        
        #64 input features, 10 ouputs features for our 10 defined classes
        self.fc2 = torch.nn.Linear(512, num_class)
        
    
    def forward(self, x):
        #Activation  of the first convolution
        #Size changes from (3, 224, 224) to (18, 224, 224)
        x = F.relu(self.conv1(x))
        first_layer = x
        #Size changes from (18, 224, 224) to (18, 112, 112)
        x = self.pool(x)
        
        #Second conv
        #Size changes from (18, 112, 112) to (36, 112, 112)
        x = F.relu(self.conv2(x))
        second_layer = x
        
        #Size changes from (36, 112, 112) to (36, 56, 56)
        x = self.pool(x)
        
        #Reshape
        #Sizes changes from (36, 56, 56) to (1, 36 * 56 * 56)
        x = x.view(-1, 18 * 52 * 52)
        
        #Activation of the first fc layer
        #Size changes from (1, 36 * 56 * 56) to (1, 512)
        x = F.relu(self.fc1(x))
        
        #Second fc
        #Size changes from (1,512) to (1, num_classes)
        x = self.fc2(x)
        #x = self.fc2(x)
        
        
        #LedDropOut
        
        return x