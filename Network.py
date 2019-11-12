import torch
import torch.nn as nn
from torch.nn import functional as F

class gqcnn(nn.Module):
    def __init__(self):
        super(gqcnn, self).__init__()
        # root 1
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=7)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=5)
        self.lrn = nn.LocalResponseNorm(2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_3 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv1_4 = nn.Conv2d(64, 64, kernel_size=3)
        self.fc1_1 = nn.Linear(64 * 7 * 7, 1024)
        #root 2
        self.fc2_1 = nn.Linear(1, 16)
        #root 3
        self.fc3_1 = nn.Linear(1040, 1024)
        self.fc3_2 = nn.Linear(1024,2)
        self.softmax = nn.Softmax(dim=1)



    def forward(self, x1, x2):
        x1 = F.relu(self.conv1_1(x1))
        x1 = self.pool(self.lrn(F.relu(self.conv1_2(x1))))
        x1 = F.relu(self.conv1_3(x1))
        x1 = self.lrn(F.relu(x1))
        x1 = self.lrn(F.relu(self.conv1_4(x1)))
        x1 = x1.view(-1, 64 * 7 * 7)
        #print(x1.shape)
        x1 = self.fc1_1(x1)
        x1 = F.relu(x1)
        #print(x1.shape)
        x2 = x2.view(-1, 1)
        #print(x2.shape)
        x2 = F.relu(self.fc2_1(x2))
        x3 = torch.cat((x1,x2),dim=1)
        x3 = F.relu(self.fc3_1(x3))
        x3 = self.softmax(self.fc3_2(x3))
        #print(x3.shape)
        return x3

class improved_gqcnn(nn.Module):
    def __init__(self):
        super(gqcnn, self).__init__()
        # root 1
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=7)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=5)
        self.lrn = nn.LocalResponseNorm(2)
        self.pool = nn.MaxPool2d(2, 2)
        self.layer1 = self._make_layer(64, 64, 2)
        self.fc1_1 = nn.Linear(64 * 5 * 5, 1024)
        #root 2
        self.fc2_1 = nn.Linear(1, 16)
        #root 3
        self.fc3_1 = nn.Linear(1040, 1024)
        self.fc3_2 = nn.Linear(1024,2)
        self.softmax = nn.Softmax(dim=1)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel))

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))

    def forward(self, x1, x2):
        x1 = F.relu(self.conv1_1(x1))
        x1 = self.pool(self.lrn(F.relu(self.conv1_2(x1))))
        x1 = self.layer1(x1)
        x1 = x1.view(-1, 64 * 5 * 5)
        #print(x1.shape)
        x1 = self.fc1_1(x1)
        x1 = F.relu(x1)
        #print(x1.shape)
        x2 = x2.view(-1, 1)
        #print(x2.shape)
        x2 = F.relu(self.fc2_1(x2))
        x3 = torch.cat((x1,x2),dim=1)
        x3 = F.relu(self.fc3_1(x3))
        x3 = self.softmax(self.fc3_2(x3))
        #print(x3.shape)
        return x3

class ResidualBlock(nn.Module):
    def __init__(self,inchannel,outchannel,stride=1,shortcut=None):
        super(ResidualBlock,self).__init__()
        self.left=nn.Sequential(
            nn.Conv2d(inchannel,outchannel,3,stride,1,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,outchannel,3,1,1,bias=False),
            nn.BatchNorm2d(outchannel)

        )

        self.right=shortcut

    def forward(self,x):
        out=self.left(x)
        residual=x if self.right is None else self.right(x)
        out+=residual
        return F.relu(out)


