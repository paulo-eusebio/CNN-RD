import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        pad = (1,1,1,1)
        x = F.pad(x, pad, 'replicate')
        return self.relu(self.conv(x))


class CNNCRluma(nn.Module):
    def __init__(self):
        super(CNNCRluma, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        #CNN
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 8)
        #self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        #self.basic = nn.ModuleList([self.conv for i in range(self.hid_depth)])
        self.conv_out = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=0)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)


    def forward(self, HR):
        #residue = get_guide(HR, factor=0.5)
        residue = F.interpolate(HR, scale_factor=0.5, mode='bicubic')
        pad = (1, 1, 1, 1)
        out = F.pad(HR, pad, 'replicate')
        out = self.conv1(out)
        out = self.relu(out)
        out = self.residual_layer(out)
        pad = (1, 1, 1, 1)
        out = F.pad(out, pad, 'replicate')
        out = self.conv_out(out)
        out = out +residue
        return out, residue

class CNNSRluma(nn.Module):
    def __init__(self):
        super(CNNSRluma, self).__init__()
        self.relu = nn.ReLU(inplace=True)  # to save memory, remove if error

        #CNN
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv2a = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv2b = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(in_channels=48, out_channels=48, kernel_size=12, stride=2, padding=5)
        self.conv4a = nn.Conv2d(in_channels=48, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(in_channels=48, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv5 = nn.Conv2d(in_channels=48, out_channels=1, kernel_size=3, stride=1, padding=1)


    def forward(self, LR):
        F1 = self.conv1(LR)
        F1 = self.relu(F1)
        F2a = self.conv2a(F1)
        F2b = self.conv2b(F1)

        F2 = self.concat_volumes(self.relu(F2a), self.relu(F2b))
        F2 = self.deconv(F2) #TODO REPLACE WITH PIXEL SHUFFLE PROLLY
        F2 = self.relu(F2)

        F4a = self.conv4a(F2)
        F4b = self.conv4b(F2)
        del F1, F2, F2a, F2b
        F4 = self.concat_volumes(self.relu(F4a), self.relu(F4b))
        F4 = self.conv5(F4)
        guide = F.interpolate(LR, scale_factor=2, mode='bicubic')
        #guide = get_guide(LR, factor=2)
        F4 = F4 + guide
        return F4

    def concat_volumes(self, volume1, volume2):
        output = [volume1, volume2]
        return torch.cat(output, 1)