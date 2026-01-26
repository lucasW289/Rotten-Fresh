import torch
import torch.nn as nn
import torch.nn.functional as F

class FruitCNN(nn.Module):
    def __init__(self):
        super(FruitCNN, self).__init__()
        
        # --- Feature Extractor (The Eyes) ---
        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) # Added Batch Norm for stability
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Max Pooling (standard)
        self.pool = nn.MaxPool2d(2, 2)
        
        # --- The Fix: Adaptive Pooling ---
        # Instead of flattening 28x28, we force it down to 4x4
        # This reduces parameters from 50 Million to ~1 Million
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # --- Classifier (The Brain) ---
        self.flatten_size = 128 * 4 * 4 # = 2048 (Much smaller!)
        
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 2) 

    def forward(self, x):
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Force the output to be 4x4 spatially
        x = self.adaptive_pool(x)
        
        # Flatten and Classify
        x = x.view(-1, self.flatten_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def build_model():
    return FruitCNN()