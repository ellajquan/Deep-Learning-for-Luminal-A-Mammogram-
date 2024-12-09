import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet(nn.Module):

    def __init__(self, num_classes=2):
        super(ResNet, self).__init__()
        self.resnet1 = resnet18(pretrained=True)
        self.resnet1.fc = nn.Linear(512, 64)
        self.resnet2 = resnet18(pretrained=True)
        self.resnet2.fc = nn.Linear(512, 64)
        self.resnet3 = resnet18(pretrained=True)
        self.resnet3.fc = nn.Linear(512, 64)
        self.resnet4 = resnet18(pretrained=True)
        self.resnet4.fc = nn.Linear(512, 64)
        self.fc = nn.Sequential(
            nn.Linear(64 * 4, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x1, x2, x3, x4):
        out1 = self.resnet1(x1)
        out2 = self.resnet2(x2)
        out3 = self.resnet3(x3)
        out4 = self.resnet4(x4)
        out = torch.cat([out1, out2, out3, out4], dim=-1)
        return self.fc(out)





































