import torch.nn as nn
from torchvision import models
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

class Ensemble(nn.Module):
    def __init__(self, modelA, modelB):
        super(Ensemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.fc1 = nn.Linear(100, 50)
        #self.fc2 = nn.Linear(75, 50)
        self.soft = nn.Softmax()
        
    
    def forward(self, x1):
        x_1 = self.modelA(x1)
        x_2 = self.modelB(x1)
        x = torch.cat((x_1,x_2), dim=1)
        x = self.fc1(x)
        #x = self.soft(x)
        return x