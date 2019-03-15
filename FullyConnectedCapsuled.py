from torch.autograd import Variable
import torch.nn.functional as F
import torch

class FullyConnectedCapsuled(torch.nn.Module):
    
    #Batch shape for input x is (3, 224, 224)
    
    def __init__(self, num_of_features, num_class = 3):
        super(FullyConnectedCapsuled, self).__init__()
        
        #self.num_class = 6

        #FC layers to new layer
        #Number of classes belong to shape triangle group
        self.fc_nl_1 = torch.nn.Linear(num_of_features, 19)
        
        #Number of classes belong to shape circle group
        self.fc_nl_2 = torch.nn.Linear(num_of_features, 18)
        
        #Number of classes belong to shape cylinder group
        self.fc_nl_3 = torch.nn.Linear(num_of_features, 6)
        
        #Number of classes belong to shape square group
        self.fc_nl_4 = torch.nn.Linear(num_of_features, 2)
        
        #Number of classes belong to shape diamond group
        self.fc_nl_5 = torch.nn.Linear(num_of_features, 1)
        
        #Number of classes belong to shape other group
        self.fc_nl_6 = torch.nn.Linear(num_of_features, 4)
        
    
    def forward(self, x):        
        #Vectors to highlight the shape group
        x_out = self.fc_nl_1(x)
        x_out = torch.cat((x_out, self.fc_nl_2(x)),1)
        x_out = torch.cat((x_out, self.fc_nl_3(x)),1)
        x_out = torch.cat((x_out, self.fc_nl_4(x)),1)
        x_out = torch.cat((x_out, self.fc_nl_5(x)),1)
        x_out = torch.cat((x_out, self.fc_nl_6(x)),1)
        return x_out

    
                #print(p1.shape)
                #plt.figure(figsize=(16,8))
                #p1 = torch.unsqueeze(p1, 2)
                #print(p1.shape)
                #grid_image = torchvision.utils.make_grid(p1[100], nrow=9)
                #print(grid_image.shape)
                #grid_image = grid_image.permute(1,2,0)
                #plt.imshow(grid_image)  
                #plt.show()
                
                #print(p2.shape)
                #plt.figure(figsize=(16,8))
                #p2 = torch.unsqueeze(p2, 2)
                #print(p2.shape)
                #grid_image = torchvision.utils.make_grid(p2[100], nrow=9)
                #print(grid_image.shape)
                #grid_image = grid_image.permute(1,2,0)
                #plt.imshow(grid_image)  
                #plt.show()