import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )
    def forward(self, inp):
        features = self.seq(inp)
        down = torch.nn.functional.max_pool2d(features, kernel_size=2)
        # Features is for concatenating with the decoder blocks
        # Down is for the next encoder block
        return features, down 

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        # Upscale and half the number of features
        self.trans_conv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2)
        self.seq = nn.Sequential(
            # so that when we concat the encoder block's features
            # the amount of input features stays the same
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )
    def forward(self, inp, con_channels):
        up = self.trans_conv(inp)

        # We need to pad the mask when we concatenating
        # upscaled features that were previously 
        # downscaled from odd dimension features
        # For example: 25 -> down -> 12 -> up -> 24 -> pad -> 25
        diffY = con_channels.size()[2] - up.size()[2]
        diffX = con_channels.size()[3] - up.size()[3]
        up = torch.nn.functional.pad(up, [diffX // 2, diffX - diffX // 2,
                                          diffY // 2, diffY - diffY // 2])
        
        cat = torch.cat([up, con_channels], dim=1)
        return self.seq(cat)
    
class LatentSpace(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )
    def forward(self, inp):
        return self.seq(inp)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, inp):
        return self.seq(inp)

class CEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.Sigmoid(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.Sigmoid(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.1)
        )
        self.down_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=2)
    def forward(self, inp):
        features = self.seq(inp)
        # Features is for concatenating with the decoder blocks
        # Down is for the next encoder block
        down = self.down_conv(features)
        return features, down

class CDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        # Upscale and half the number of features
        self.trans_conv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2)
        self.seq = nn.Sequential(
            # so that when we concat the encoder block's features
            # the amount of input features stays the same
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.Sigmoid(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.Sigmoid(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.1)
        )
    def forward(self, inp, con_channels):
        up = self.trans_conv(inp)

        # We need to pad the mask when we concatenating
        # upscaled features that were previously 
        # downscaled from odd dimension features
        # For example: 25 -> down -> 12 -> up -> 24 -> pad -> 25
        diffY = con_channels.size()[2] - up.size()[2]
        diffX = con_channels.size()[3] - up.size()[3]
        up = torch.nn.functional.pad(up, [diffX // 2, diffX - diffX // 2,
                                          diffY // 2, diffY - diffY // 2])
        
        cat = torch.cat([up, con_channels], dim=1)
        return self.seq(cat)
    
class ExtendedConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=2, stride=1, dilation=2):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
            nn.Sigmoid(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
            nn.Sigmoid(),
            nn.BatchNorm2d(out_channels),

            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
            nn.Sigmoid(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
            nn.Sigmoid(),
            nn.BatchNorm2d(out_channels),

            nn.Dropout2d(0.1)
        )
    def forward(self, inp):
        return self.seq(inp)

class COutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.Tanh()
        )
    def forward(self, inp):
        return self.seq(inp)
    
#https://en.wikipedia.org/wiki/Huber_loss
class HuberLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target, delta=1):
        diff = torch.abs(pred - target)
        smooth_l1_loss = 0.5 * torch.where(diff < delta, diff ** 2, delta * (diff - 0.5 * delta))
        return torch.mean(smooth_l1_loss)