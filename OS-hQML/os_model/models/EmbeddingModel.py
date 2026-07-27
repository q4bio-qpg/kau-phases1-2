import torch
from torch import nn
import torch.nn.functional as F

class ConvEmbeddingModel(nn.Module):
  def __init__(self, input_size):
        super(ConvEmbeddingModel, self).__init__()
        self.flatten = nn.Flatten()
        self.init_conversion_conv = nn.Conv1d(1,1,kernel_size=2,stride=2)
        self.patternt_searching_conv = nn.Conv1d(1,1,kernel_size=4,stride=1)
        self.output_size = int(input_size / 2 - 3)

  def forward(self, x):
    x = self.flatten(x)
    x = x.unsqueeze(1)
    conv_sequence = self.init_conversion_conv(x)
    pattern_sequence = self.patternt_searching_conv(conv_sequence)
    return pattern_sequence

class EditDistanceModel(nn.Module):

    def __init__(self, out_dim=200):
        super(EditDistanceModel, self).__init__()
        self.out_dim = out_dim
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv1d(1, 1, kernel_size=2, stride=1, padding=0) 
        self.bn1 = nn.BatchNorm1d(1)
        self.flatten_dim = 63
        self.fc1 = nn.Linear(self.flatten_dim, 1024)
        self.fc3 = nn.Linear(1024, self.out_dim)
    
    def forward(self, x):
       
        x = self.flatten(x)
        
        x = x.unsqueeze(1) 
       
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.flatten(x)
        
        x = F.relu(self.fc1(x))
       
        x = self.fc3(x)
       
        x = F.normalize(x, dim=-1)
        
        return x