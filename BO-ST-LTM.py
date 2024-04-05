import torch
import os
import h5py
import cv2
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torch.nn.functional as F
from torch.optim import Adam
from torchmetrics import MeanSquaredError
import torchvision.transforms as transforms
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from sklearn.model_selection import KFold

model_type = "GP"
acquisition_type = "MPI"

#MPI
best_conf = [0.34112841283763873,4,16,4,16,1,5,0.001]

best_path = "/home/wyl123/csp-drive-dl/data/def/run_2/gpyopt/GP_MPI/gpyopt_model_run_15.h5"
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def rmse_loss(outputs, labels):
    return torch.sqrt(F.mse_loss(outputs, labels))

def conf_dir(env_key, default_value):
    return os.path.expanduser(os.getenv(env_key, default_value))

data_dir = conf_dir('PC_DATA_DIR', "~/csp-drive-dl/data/def")

class ConvLSTM(nn.Module):
    def __init__(self, input_shape, l1_out=16, l2_out=4, l3_out=16, l4_out=16, l5_out=1, l6_out=50, l1_drop=0.4692891197930165, lr_adam=0.001):
        super(ConvLSTM, self).__init__()
        self.conv_lstm1 = nn.Conv2d(input_shape[0], l1_out, kernel_size=(3, 3), padding=1)
        self.lstm1 = nn.LSTM(input_size=l1_out, hidden_size=l1_out, num_layers=1, batch_first=True)
        self.bn1 = nn.BatchNorm2d(l1_out)
        self.conv_lstm2 = nn.Conv2d(l1_out, l2_out, kernel_size=(3, 3), padding=1)
        self.lstm2 = nn.LSTM(input_size=l2_out, hidden_size=l2_out, num_layers=1, batch_first=True)
        self.bn2 = nn.BatchNorm2d(l2_out)
        self.conv_lstm3 = nn.Conv2d(l2_out, l3_out, kernel_size=(3, 3), padding=1)
        self.lstm3 = nn.LSTM(input_size=l3_out, hidden_size=l3_out, num_layers=1, batch_first=True)
        self.bn3 = nn.BatchNorm2d(l3_out)
        self.conv_lstm4 = nn.Conv2d(l3_out, l4_out, kernel_size=(3, 3), padding=1)
        self.lstm4 = nn.LSTM(input_size=l4_out, hidden_size=l4_out, num_layers=1, batch_first=True)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv3d = nn.Conv3d(16, l5_out, kernel_size=(3, 3, 3), padding=1)
       
        self.max_pool3d = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(281600, l6_out)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=-0.2)
        self.dropout = nn.Dropout(l1_drop)
        self.fc2 = nn.Linear(l6_out, 1)

    def forward(self, x):
        
        x = x / 255.0
        batch_size, seq_len, channels, height, width = x.size()
#        print("After reshape:", x.size())
#        x = x.permute(0, 2, 1, 3, 4)
       
    
        x = x.reshape(batch_size * seq_len, channels, height, width)

#        print("After reshape:", x.size())
        
        x = self.conv_lstm1(x)
#        print("After conv_lstm1:", x.size())
#        x = self.bn1(x)
#        print("After bn1:", x.size())
        x = self.conv_lstm2(x)
#        print("After conv_lstm2:", x.size())
        x = self.bn2(x)
#        print("After bn2:", x.size())
        x = self.conv_lstm3(x)  
#        print("After conv_lstm3:", x.size())
        x = self.bn3(x)
#        print("After bn3:", x.size())
        x = self.conv_lstm4(x)
#        print("After conv_lstm4:", x.size())
        x = self.bn4(x)
#        print("After bn4:", x.size())
#        print(x.shape)
        
        x = x.unsqueeze(0)
#        print(x.shape)
#        x = self.conv3d(x)
       

#        print("After conv3d:", x.size())
        x = self.max_pool3d(x)
#        print("After max_pool3d:", x.size())
        x = torch.flatten(x, start_dim=1)
#        print("After flatten:", x.size())
        x = self.fc1(x)
#        print("After fc1:", x.size())
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
#        print("After fc2:", x.size())
    
        
    
        return x    
    def configure_optimizers(self):
       optimizer = optim.Adam(self.parameters(), lr=0.001)
       scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.9 ** (epoch // 5))
       return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
#        loss = nn.MSELoss()(y_hat, y.view(-1, 1).float())
        loss = torch.sqrt(F.mse_loss(y_hat, y.view(-1, 1).float()))
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
#        loss = nn.MSELoss()(y_hat, y.view(-1, 1).float())
        loss = torch.sqrt(F.mse_loss(y_hat, y.view(-1, 1).float()))
        
        self.log('val_loss', loss)