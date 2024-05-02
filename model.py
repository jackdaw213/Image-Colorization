import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import ResNet34_Weights
from torchvision.models import VGG19_Weights
import model_parts as mp
import torch.nn.functional as F

"""
So the old artistic* style transfer model is probably in a separate repo now.
The reason is because the reading comprehension curse got me and I didn't realize
that the style transfer model that the researchers in the main paper of this repo
were using was photorealistic* style transfer instead of the artistic one. But
there are a few problems, first, the way the style transfer model works was not 
really described in detail by them making it really hard to implement. This is also
the reason why I implemented the artistic style model from the AdaIn paper instead
of the main one (I didn't know that artistic and photorealistic models are completely
different). Second the code for the main paper is not runnable, the transfer model
uses VGGDecoder and VGGEncoder classes and their definitions are nowhere to be found.
Until I looked at the repo for PhotoWCT paper and found the definitions for both. 
So apparently they used the code from the PhotoWCT paper but didn't mention it anywhere
But they did foreshadow it by comparing the results of PhotoAdaIN (their style model)
with PhotoWCT multiple times in the paper (I lack vision to see it)

But a new problem, PhotoWCT uses multi-level stylization and it's EXTREMELY complicated
to understand and implement. So I need to find another photorealistic model that doesn't 
hit my brain as hard. I stumbled upon WCT^2, they use progressive colorization (as they 
called it), and it's way less complicated. I was about to settle for it until I discovered 
this [1*]. This paper is amazing, they gather all of the photorealistic style transfer 
methods compare them with eachother to find the best one, insanely informative. The best 
thing is that it's the simplest method I found and I decided to implement it until I hit 
a roadblock and had to look at their code base. Saying it's messy is an understatement 
but to their credit, the P-Step mentioned in the paper is simply that complicated but I 
think that they could have done better with the code duplication

[1*]: https://arxiv.org/pdf/1912.02398
"""

class StyleTransfer(nn.Module):
    def __init__(self):
        super().__init__()

        vgg19 = torchvision.models.vgg19(weights=VGG19_Weights.DEFAULT).features

        for param in vgg19.parameters():
            param.requires_grad = False

        # ReLU 1_1 -> ReLu 5_1
        self.e1 = nn.Sequential(*list(vgg19.children())[:2])
        self.e2 = nn.Sequential(*list(vgg19.children())[2:7])
        self.e3 = nn.Sequential(*list(vgg19.children())[7:12])
        self.e4 = nn.Sequential(*list(vgg19.children())[12:21])
        self.e5 = nn.Sequential(*list(vgg19.children())[21:30])

        self.in1 = nn.InstanceNorm2d(64)
        self.in2 = nn.InstanceNorm2d(128)
        self.in3 = nn.InstanceNorm2d(256)
        self.in4 = nn.InstanceNorm2d(512)

        self.max1 = nn.MaxPool2d(kernel_size=16,stride=16)
        self.max2 = nn.MaxPool2d(kernel_size=8,stride=8)
        self.max3 = nn.MaxPool2d(kernel_size=4,stride=4)
        self.max4 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.wct = mp.WCT()

        self.d5 = mp.VggDecoderBlock(512, 512, 5)
        self.d4 = mp.VggDecoderBlock(512, 256, 4)
        self.d3 = mp.VggDecoderBlock(256, 128, 3)
        self.d2 = mp.VggDecoderBlock(128, 64, 2)
        self.d1 = mp.VggDecoderBlock(64, 3, 1)


    def encoder(self, input):
        features = []

        out = self.e1(input)
        features.append(out)

        out = self.e2(out)
        features.append(out)

        out = self.e3(out)
        features.append(out)

        out = self.e4(out)
        features.append(out)

        out = self.e5(out)

        return out, features
    
    def bfa(self, input, concat):

        # They use instance norm before concatenating in the code base even though
        # this was not mentioned in the paper
        input = torch.concat([input, self.max1(self.in1(concat[0]))], dim=1)
        input = torch.concat([input, self.max2(self.in2(concat[1]))], dim=1)
        input = torch.concat([input, self.max3(self.in3(concat[2]))], dim=1)
        input = torch.concat([input, self.max4(self.in4(concat[3]))], dim=1)

        return input
    
    def insl(self, content_features, style_features):
        insl_features = []

        insl_features.append(self.wct(content_features[0], style_features[0]))
        insl_features.append(self.wct(content_features[1], style_features[1]))
        insl_features.append(self.wct(content_features[2], style_features[2]))
        insl_features.append(self.wct(content_features[3], style_features[3]))
        
        return insl_features

    def forward(self, content, style=None):
        # Training mode, no wct
        if style is None:
            content, content_features = self.encoder(content)
            content_features.append(content)

            content = self.bfa(content, content_features)

            content = self.d5(content, None)
            content = self.d4(content, self.in4(content_features[3]))
            content = self.d3(content, self.in3(content_features[2]))
            content = self.d2(content, self.in2(content_features[1]))
            content = self.d1(content, self.in1(content_features[0]))

            _, content_features_loss = self.encoder(content)
            content_features_loss.append(_)

            return content, content_features, content_features_loss
        
        content, content_features = self.encoder(content)
        style, style_features = self.encoder(style)

        insl = self.insl(content_features, style_features)
        content_wct = self.wct(content, style)

        content_wct = self.bfa(content_wct, insl)
        style = self.bfa(style, style_features)

        content_wct = self.d5(content_wct, None)
        style = self.d5(style, None)
        content_wct = self.wct(content_wct, style)

        content_wct = self.d4(content_wct, self.in4(insl[3]))
        style = self.d4(style, self.in4(style_features[3]))
        content_wct = self.wct(content_wct, style)

        content_wct = self.d3(content_wct, self.in3(insl[2]))
        style = self.d3(style, self.in3(style_features[2]))
        content_wct = self.wct(content_wct, style)

        content_wct = self.d2(content_wct, self.in2(insl[1]))
        style = self.d2(style, self.in2(style_features[1]))
        content_wct = self.wct(content_wct, style)

        content_wct = self.d1(content_wct, self.in1(insl[0]))

        return content_wct
    
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

        self.ls = mp.LatentSpace(512, 512)

        self.d6 = mp.DecoderBlock(512, 256)
        self.d5 = mp.DecoderBlock(256, 128)
        self.d4 = mp.DecoderBlock(128, 64)
        self.d3 = mp.DecoderBlock(64, 64)
        self.d2 = mp.DecoderBlock(64, 32)
        self.d1 = mp.DecoderBlock(32, 32, 3)

        self.out = mp.OutConv(32, 2)

    def forward(self, x):
        inp = x
        features_dict = {}

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
        x = self.d2(x, features_dict["layer0"]) # 7x7 convolution
        x = self.d1(x, inp)

        return self.out(x)
    
    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.e1 = mp.EncoderBlock(1, 64)
        self.e2 = mp.EncoderBlock(64, 128)
        self.e3 = mp.EncoderBlock(128, 256)
        self.e4 = mp.EncoderBlock(256, 512)

        self.ls = mp.LatentSpace(512, 1024)

        self.d4 = mp.DecoderBlock(1024, 512)
        self.d3 = mp.DecoderBlock(512, 256)
        self.d2 = mp.DecoderBlock(256, 128)
        self.d1 = mp.DecoderBlock(128, 64)

        self.out = mp.OutConv(64, 2)

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
