import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import ResNet34_Weights
from torchvision.models import VGG19_Weights
import auto_parts as ap
import torch.nn.functional as F

class StyleTransfer(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg19 = torchvision.models.vgg19(weights=VGG19_Weights.DEFAULT).features

        for param in self.vgg19.parameters():
            param.requires_grad = False

        self.vgg19_style_map = {
            '1': 'relu1_1',
            '6': 'relu2_1',
            '11': 'relu3_1', 
            '20': 'relu4_1'
        }

        self.vgg19_concat_map = {
            '3': 'relu1_2',
            '8': 'relu2_2',
            '17': 'relu3_4'
        }

        self.adain = ap.AdaIN()

        # There is no need to concatenate features for layer 4 so upscale is not
        # going to havle the number of features
        self.d4 = ap.VggDecoderBlock(512, 256, 4)
        self.d3 = ap.VggDecoderBlock(256, 128, 3)
        self.d2 = ap.VggDecoderBlock(128, 64, 2)
        self.d1 = ap.VggDecoderBlock(64, 2, 1)

    def forward(self, content, style, encoder_only=False):
        layer = 1
        concat_features = {}
        style_features = []

        # Cut off the model and sets up hooks is cleaner than looping through modules
        # But I want to use torch.compile(), which doesn't support hooks
        for num, module in self.vgg19.named_modules():
            if num != '' and int(num) <= 20:
                # In the paper the author replaces max pool with avg pool
                if isinstance(module, nn.MaxPool2d):
                    content = F.avg_pool2d(content, kernel_size=2, stride=2)
                    style = F.avg_pool2d(style, kernel_size=2, stride=2)
                else:
                    content = module(content)
                    style = module(style)

                if num in self.vgg19_style_map:
                    style_features.append(style)

                if num in self.vgg19_concat_map and not encoder_only:
                    concat_features[f"layer{layer}"] = content
                    layer += 1

        adain = self.adain(content, style)

        if not encoder_only:
            x = self.d4(adain, None)
            print(x.shape)
            print(concat_features["layer3"].shape)
            x = self.d3(x, concat_features["layer3"])
            print(x.shape)
            print(concat_features["layer2"].shape)
            x = self.d2(x, concat_features["layer2"])
            x = self.d1(x, concat_features["layer1"])
            
            return x, adain, style_features

        return content, adain, style_features

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