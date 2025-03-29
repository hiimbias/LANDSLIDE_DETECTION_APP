import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.encoder1 = ConvBlock(in_channels, 8)
        self.encoder2 = ConvBlock(8, 16)
        self.encoder3 = ConvBlock(16, 32)
        self.encoder4 = ConvBlock(32, 64)
        self.encoder5 = ConvBlock(64, 128)

        self.pool = nn.MaxPool2d(2)

        self.upconv2 = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.upconv5 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.decoder1_1 = ConvBlock(16, 8)#x2
        self.decoder2_1 = ConvBlock(32, 16)
        self.decoder3_1 = ConvBlock(64, 32)
        self.decoder4_1 = ConvBlock(128, 64)

        self.decoder1_2 = ConvBlock(24, 8)#x3
        self.decoder2_2 = ConvBlock(48, 16)
        self.decoder3_2 = ConvBlock(96, 32)

        self.decoder1_3 = ConvBlock(32, 8)#x4
        self.decoder2_3 = ConvBlock(64, 16)

        self.decoder1_4 = ConvBlock(40, 8)#x5

        self.final_conv = nn.Conv2d(8, num_classes, kernel_size=1)


    def forward(self, x):
        x_00 = self.encoder1(x)
        x_10 = self.encoder2(self.pool(x_00))
        x_20 = self.encoder3(self.pool(x_10))
        x_30 = self.encoder4(self.pool(x_20))
        x_40 = self.encoder5(self.pool(x_30))

        x_01 = self.decoder1_1(torch.cat([x_00, self.upconv2(x_10)], dim=1))
        x_11 = self.decoder2_1(torch.cat([x_10, self.upconv3(x_20)], dim=1))
        x_21 = self.decoder3_1(torch.cat([x_20, self.upconv4(x_30)], dim=1))
        x_31 = self.decoder4_1(torch.cat([x_30, self.upconv5(x_40)], dim=1))

        x_02 = self.decoder1_2(torch.cat([x_00, x_01, self.upconv2(x_11)], dim=1))
        x_12 = self.decoder2_2(torch.cat([x_10, x_11, self.upconv3(x_21)], dim=1))
        x_22 = self.decoder3_2(torch.cat([x_20, x_21, self.upconv4(x_31)], dim=1))

        x_03 = self.decoder1_3(torch.cat([x_00, x_01, x_02, self.upconv2(x_12)], dim=1))
        x_13 = self.decoder2_3(torch.cat([x_10, x_11, x_12, self.upconv3(x_22)], dim=1))

        x_04 = self.decoder1_4(torch.cat([x_00, x_01, x_02, x_03, self.upconv2(x_13)], dim=1))

        return self.final_conv(x_04)