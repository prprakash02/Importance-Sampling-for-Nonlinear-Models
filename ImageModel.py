import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

image_size = 10

class SimpleNNBinary(nn.Module):
    def __init__(self):
        super(SimpleNNBinary, self).__init__()
        self.fc1 = nn.Linear(image_size * image_size, image_size, bias=False)  # 10x10 input features
        self.fc2 = nn.Linear(image_size, 1, bias=False) # 10 hidden neurons and output

    def forward(self, x):
        x = x.view(-1, image_size * image_size)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x    
