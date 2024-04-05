import torch
import torch.nn as nn
import torch.nn.functional as F
from ncps.wirings import AutoNCP
from ncps.torch import LTC
from torchvision import models

def configure_optimizers(self):
    optimizer = optim.Adam(self.parameters(), lr=0.001)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.9 ** (epoch // 5))
    return [optimizer], [scheduler]

class PilotNet(LightningModule):
    def __init__(self, input_shape):
        super(PilotNet, self).__init__()
        self.conv1 = nn.Conv2d(160, 24, kernel_size=5, stride=2, padding=2, bias=True)
      
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=2, bias=True)
        
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=2, bias=True)
        
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
      
        self.fc1 = nn.Linear(2560, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        x = x.float() / 255.0
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        x = x.view(x.size(0), -1)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
#        print("Target (y):", y)
#        print("Prediction (y_hat):", y_hat)
#        loss = F.mse_loss(y_hat, y)
        loss = torch.sqrt(F.mse_loss(y_hat, y.unsqueeze(1).float()))
        self.log('train_loss', loss)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
#        loss = nn.MSELoss()(y_hat, y.view(-1, 1).float())
        loss = torch.sqrt(F.mse_loss(y_hat, y.unsqueeze(1).float()))
        self.log('val_loss', loss)
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        print("Target (y):", y)
        print("Prediction (y_hat):", y_hat)
        loss = torch.sqrt(F.mse_loss(y_hat, y.unsqueeze(1).float())) 
        self.log('test_loss', loss)  
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)