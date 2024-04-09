import torch
import torch.nn as nn
import torch.nn.functional as F
from ncps.wirings import AutoNCP
from ncps.torch import LTC
from torchvision import models



class SteeringAnglePredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.conv1 = nn.Conv2d(input_dim, 24, kernel_size=2)
        self.fc1 = nn.Linear(input_dim, 32)
        self.dropout = nn.Dropout(0.25)
        self.ltc = LTC(32, wiring, batch_first=True)
    
    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = x.view(x.size(0), -1, 32)
        x = x.squeeze(2)
        x = x.view(x.size(0), x.size(1), x.size(2))
        x = x.mean(1)
        x = self.ltc(x)[0]
        return x


class BinaryMaskGenerator(nn.Module):
    def __init__(self):
        super(BinaryMaskGenerator, self).__init__()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, seg_output):
        mask = self.sigmoid(seg_output)
        return mask
