import torch
import torch.nn as nn
import torch.nn.functional as F

class FruitCNN(nn.Module):
    def __init__(self):
        super(FruitCNN, self).__init__()
        
        # 1. Convolutional Layers
        # Input: 3 Channels (RGB) -> Output: 32 Feature Maps
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 2. Pooling Layer (Max Pool 2x2)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 3. Fully Connected Layers
        # Calculation: 224 -> 112 -> 56 -> 28 (after 3 pools)
        self.flatten_size = 128 * 28 * 28
        
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 2) # [Fresh, Rotten]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(-1, self.flatten_size) # Flatten
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def build_model():
    return FruitCNN()