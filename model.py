import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import ResNet34_Weights
import auto_parts as ap
import torch.nn.functional as F

class UNetResEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet.resnet34(weights=ResNet34_Weights.DEFAULT)

        for param in resnet.parameters():
            param.requires_grad = False

        resnet_layers = []
        resnet_layers.append(nn.Sequential(*list(resnet.children())[:3]))

        for i, layer in enumerate(resnet.children(), 1):
            if isinstance(layer, nn.Sequential):
                resnet_layers.append(layer)

        # Need to wrap this in a module list or else cuda() will not work
        self.resnet_layers = nn.ModuleList(resnet_layers)

        self.ls = ap.LatentSpace(512, 1024)

        self.d6 = ap.DecoderBlock(1024, 512)
        self.d5 = ap.DecoderBlock(512, 256)
        self.d4 = ap.DecoderBlock(256, 128)
        self.d3 = ap.DecoderBlock(128, 64)
        self.d2 = ap.DecoderBlock(in_channels=32 + 64, out_channels=64, 
                                  up_in_channels=64, up_out_channels=32)
        self.d1 = ap.DecoderBlock(in_channels=32 + 1, out_channels=64, 
                                  up_in_channels=64, up_out_channels=32)

        self.out = ap.OutConv(64, 2)

    def forward(self, x):
        inp = x
        features_dict = {}

        # Input is gray scale image with 1 channel, resnet needs 3 so we need
        # to pad the image with extra 2 channels of 0
        x = nn.functional.pad(x, (0, 0, 0, 0, 1, 1), mode='constant', value=0)

        for i, layer in enumerate(self.resnet_layers, 0):
            x = layer(x)
            features_dict[f"layer{i}"] = x
            if i == 0:
                x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        x = nn.functional.max_pool2d(x, kernel_size=2)
        x = self.ls(x)

        x = self.d6(x, features_dict["layer4"])
        x = self.d5(x, features_dict["layer3"])
        x = self.d4(x, features_dict["layer2"])
        x = self.d3(x, features_dict["layer1"])
        x = self.d2(x, features_dict["layer0"])
        x = self.d1(x, inp)

        return self.out(x)
    
    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = ap.EncoderBlock(1, 64)
        self.e2 = ap.EncoderBlock(64, 128)
        self.e3 = ap.EncoderBlock(128, 256)
        self.e4 = ap.EncoderBlock(256, 512)

        self.ls = ap.LatentSpace(512, 1024)

        self.d4 = ap.DecoderBlock(1024, 512)
        self.d3 = ap.DecoderBlock(512, 256)
        self.d2 = ap.DecoderBlock(256, 128)
        self.d1 = ap.DecoderBlock(128, 64)

        self.out = ap.OutConv(64, 2)

    def forward(self, x):
        r1_e_f, x = self.e1(x)
        r2_e_f, x = self.e2(x)
        r3_e_f, x = self.e3(x)
        r4_e_f, x = self.e4(x)

        x = self.ls(x)

        x = self.d4(x, r4_e_f)
        x = self.d3(x, r3_e_f)
        x = self.d2(x, r2_e_f)
        x = self.d1(x, r1_e_f)

        return self.out(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')