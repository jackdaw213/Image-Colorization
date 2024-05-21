import torch.nn as nn
import torchvision
from torchvision.models.resnet import ResNet34_Weights
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

Edit: Both style transfer models are now in their own repository, making it much easier
to manage the codebase.

[1*]: https://arxiv.org/pdf/1912.02398
"""
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

        self.d4 = mp.DecoderBlock(512, 512)
        self.d3 = mp.DecoderBlock(256, 256)
        self.d2 = mp.DecoderBlock(128, 128)
        self.d1 = mp.DecoderBlock(64, 64)

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
