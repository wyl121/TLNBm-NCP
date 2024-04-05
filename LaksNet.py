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



class LaksNet(pl.LightningModule):
    def __init__(self):
        super(LaksNet, self).__init__()
        # Define convolutional layers
        self.conv1 = nn.Conv2d(160, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        
        # Define max pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False, padding=(0, 1))
        
        # Define dropout layers
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.25)
        
        # Define fully connected layers
        self.fc1 = nn.Linear(1216, 256)
        self.fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        x = x.float()
        # Forward pass through convolutional layers
        x = self.conv1(x)
        
        x = self.pool(F.relu(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        # Flatten the output for fully connected layers
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        
        # Forward pass through fully connected layers
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
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