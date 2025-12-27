import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self, activation_type='relu',  num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.bn_fc1 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn_fc2 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, num_classes)
        self.dropout = nn.Dropout(0.5)
        if activation_type == 'relu':
            self.activation = F.relu
        elif activation_type == 'sigmoid':
            self.activation = F.sigmoid
        else:
            raise ValueError("Unsupported activation type. Use 'relu' or 'sigmoid'.")

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.activation(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 16 * 5 * 5)
        x = self.dropout(self.activation(self.bn_fc1(self.fc1(x))))
        x = self.dropout(self.activation(self.bn_fc2(self.fc2(x))))
        x = self.fc3(x)
        return x