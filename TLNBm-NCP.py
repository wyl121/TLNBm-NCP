from torchsummary import summary
import os
import h5py
import torch
import cv2
import numpy as np
import random
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from skimage.measure import label, regionprops
from ncps.wirings import AutoNCP
from ncps.torch import LTC
from model import TwinLite as net
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import StepLR

wiring = AutoNCP(19, 1)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))
class SteeringAnglePredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.conv1 = nn.Conv2d(input_dim, 8, kernel_size=2)
        self.fc1 = nn.Linear(5104, 64)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(64, 32)
        self.ltc = LTC(32, wiring, batch_first=True)
    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
#        x = x.view(x.size(0), -1)
        x = self.fc2(x)
#        x = x.view(x.size(0), -1, 32)
#        x = x.squeeze(2)
#        x = x.view(x.size(0), x.size(1), x.size(2))
#        x = x.mean(1)
        print("fc2 output size:", x.size())
        x = self.ltc(x)[0]
#        print("ltc output size:", x.size())
        return x


class BinaryMaskGenerator(nn.Module):
    def __init__(self):
        super(BinaryMaskGenerator, self).__init__()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, seg_output):
        mask = self.sigmoid(seg_output)
        return mask
