import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1)
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=1),
            nn.BatchNorm2d(output_dim)
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)

class BottomResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BottomResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1)
        )

    def forward(self, x):

        return self.conv_block(x) + x

class ResUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(ResUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 最下层先DoubleConv再snl 
        # self.inc = DoubleConv(n_channels, 4)
        # self.down1 = nn.MaxPool2d(2)
        # self.conv1 = ResidualConv(4, 8)
        # self.down2 = nn.MaxPool2d(2)
        # self.conv2 = ResidualConv(8, 16)
        # self.down3 = nn.MaxPool2d(2)
        # self.conv3 = ResidualConv(16, 32)
        # self.down4 = nn.MaxPool2d(2)
        # self.conv4 = BottomResidualConv(32, 32)
        #
        # self.up1 = Up(64, 16, bilinear)
        # self.up2 = Up(32, 8, bilinear)
        # self.up3 = Up(16, 4, bilinear)
        # self.up4 = Up(8, 8, bilinear)
        # self.outc = OutConv(8, n_classes)

        # bigger version
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = nn.MaxPool2d(2)
        self.conv1 = ResidualConv(32, 64)
        self.down2 = nn.MaxPool2d(2)
        self.conv2 = ResidualConv(64, 128)
        self.down3 = nn.MaxPool2d(2)
        self.conv3 = ResidualConv(128, 256)
        self.down4 = nn.MaxPool2d(2)
        self.conv4 = BottomResidualConv(256, 256)

        self.up1 = Up(512, 128, bilinear)
        self.up2 = Up(256, 64, bilinear)
        self.up3 = Up(128, 32, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

        # self.inc = DoubleConv(n_channels, 16)
        # self.down1 = Down(16, 32)
        # self.down2 = Down(32, 64)
        # self.down3 = Down(64, 128)
        # self.down4 = Down(128, 128)
        # self.snl = SNL(128, 4)
        # self.up1 = Up(256, 64, bilinear)
        # self.up2 = Up(128, 32, bilinear)
        # self.up3 = Up(64, 16, bilinear)
        # self.up4 = Up(32, 16, bilinear)
        # self.outc = OutConv(16, n_classes)

        # self.inc = DoubleConv(n_channels, 2)
        # self.down1 = Down(2, 4)
        # self.down2 = Down(4, 8)
        # self.down3 = Down(8, 16)
        # self.down4 = Down(16, 16)
        # self.snl = SNL(16, 4)
        # self.up1 = Up(32, 8, bilinear)
        # self.up2 = Up(16, 4, bilinear)
        # self.up3 = Up(8, 2, bilinear)
        # self.up4 = Up(4, 4, bilinear)
        # self.outc = OutConv(4, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1_ = self.down1(x1)
        x2 = self.conv1(x1_)
        x2_ = self.down2(x2)
        x3 = self.conv2(x2_)
        x3_ = self.down3(x3)
        x4 = self.conv3(x3_)
        x4_ = self.down4(x4)
        x5 = self.conv4(x4_)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

if __name__ == '__main__':
    # x = torch.rand([1, 3, 512, 512])
    net = ResUNet(n_channels=3, n_classes=2)
    # out = net(x)
    # print(net, out.shape)