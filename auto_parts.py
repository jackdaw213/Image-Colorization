import torch
import torch.nn as nn
import utils

class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        temp = torch.std(y, dim=(2, 3)) * ((x.permute([2,3,0,1]) - torch.mean(x, dim=(2, 3))) / torch.std(x, dim=(2, 3))) + torch.mean(y, dim=(2, 3))
        return temp.permute([2,3,0,1])
    
class AdaINLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass

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
    def __init__(self, in_channels, out_channels, up_in_channels=None, up_out_channels=None, kernel_size=3, padding=1):
        super().__init__()
        if up_in_channels is None or up_out_channels is None:
            up_in_channels = in_channels
            up_out_channels = in_channels // 2 # Upscale and halve the number of features
        self.trans_conv = nn.ConvTranspose2d(up_in_channels, up_out_channels, kernel_size=3, stride=2)
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
        up = utils.pad_fetures(up, con_channels)
        cat = torch.cat([up, con_channels], dim=1)
        return self.seq(cat)
    
class VggDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layer):
        super().__init__()

        # In the paper the author uses upsampling instead of conv transpose
        self.up_scale = nn.Upsample(scale_factor=2, mode='nearest')

        # So the default decoder above uses transpose which halve the number of channels
        # But we use upsample here so the number of input channels needs to be doubled except 
        # for layer 4 since we do not concatenate anything at this layer
        if layer == 1 or layer == 2:
            self.seq = nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                nn.ReLU() 
            )
        elif layer == 3:
            self.seq = nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                nn.ReLU(),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                nn.ReLU(),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                nn.ReLU(),
            )
        else: # Layer 4
            self.seq = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                nn.ReLU(),
            )
    def forward(self, inp, con_channels):
        inp = self.up_scale(inp)
        if con_channels is not None:
            inp = utils.pad_fetures(inp, con_channels)
            inp = torch.cat([inp, con_channels], dim=1)
        return self.seq(inp)
    
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