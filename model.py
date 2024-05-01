import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import ResNet34_Weights
from torchvision.models import VGG19_Weights
import model_parts as mp
import torch.nn.functional as F

"""
Today is 04/04/2024 and I'm thinking about the paper I'm trying to implement. 
I see that it was published in 2017 and think to myself: "Ha, this paper was 
published quite recently" only for a truck to hit my brain and realized that 
2017 was 7 FREAKING years ago, WTF i thought it was 3-4 years ago. Where did 
all of those years go?. I was in 8th grade and now I'm in second year of 
college ?? I swear I probably got into an accident and in coma for 3 years 
or something cause I was 18 YESTERDAY and now I'm 20!! 
I'm probably retiring next year if this keep happening :((
⣿⣿⣿⠛⢻⡟⠛⠛⠛⠛⠛⣿⠋⢻⣿⠟⠛⠋⠛⢿⣿⣿⣿⣿⣟⠛⢻⣿⡿⠛⠛⠛⠛⢿⣿⡟⠛⠛⠛⠛⡟⠛⢿⣿⣿⠛⢛⡿⠛⠛⠛⠛⣿⠛⠛⠛⠛⠻⣿⣿
⣿⣿⣿⠀⢸⣷⣷⡇⠀⣾⣾⣿⣤⣽⣇⠀⠰⢿⣷⣿⣿⣿⣿⣿⣗⠀⢸⡟⠀⢰⣾⣶⡆⠀⢻⡇⠀⢼⢾⢾⣷⠀⠘⣿⡏⠀⣼⣯⠀⢰⢷⢷⣿⠂⠀⣿⡆⠀⣸⣿
⣿⣿⣿⠀⢸⣿⣿⡇⠀⣿⣿⣿⣿⣿⣿⣦⣤⡀⠈⢻⣿⣿⣿⣿⣗⠀⢸⡇⠀⢻⣿⣿⡯⠀⢸⡇⠀⢤⣤⣼⣿⣧⠀⢻⠁⢰⣿⡷⠀⢠⣤⣬⣿⠂⠀⣀⠀⠲⣿⣿
⣿⣿⣿⠀⢸⣿⣿⡇⠀⣿⣿⣿⣿⣿⡇⠉⠛⠉⢀⣼⣿⣿⡯⠙⠁⢀⣾⣿⣄⡀⠉⠋⢀⣠⣿⡇⠀⠉⠋⠋⣿⣿⡄⠀⢀⣿⣿⣟⠀⠈⠋⠋⣿⠂⠀⣿⣧⠀⠹⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢿⢻⢕⣗⣗⢖⢍⠝⠽⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢿⣿⣿⢿⣿⡿⣿⣻⣿⣻⡽⡛⡜⡜⡮⡳⣳⢕⡯⡮⡪⡂⡊⢳⣿⣿⣽⣿⣟⣿⣿⣻⣿⣻⣿⣟⣿⣿⣻⣿⡿⣿⣿⢿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⢿⣿⣿⣽⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢏⢎⢎⢮⡳⡽⡽⣺⡳⡧⡳⣕⢕⡪⡐⢽⡿⣟⣿⣿⢿⣿⣿⣿⣿⣿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷
⣿⣿⣿⡿⣿⣽⣷⣿⣿⣿⣿⣿⣿⣿⢿⣻⣽⣿⣯⣿⣿⣽⣾⣜⢵⢝⡷⡽⣺⣝⢮⢏⡯⡯⣮⣳⢨⢈⢦⣿⣿⣿⣿⣿⣿⣷⣿⣷⣿⣿⣿⣷⣿⣿⣾⣿⣿⣻⣿⣿
⣾⣿⣷⣿⣿⣿⣿⣿⣿⣿⣿⣾⣿⣾⣿⣿⣿⡿⣟⣿⣽⣿⡻⣜⢕⡯⡮⣟⡯⣯⡳⣝⣞⢽⢕⢧⣓⢅⣿⣿⣯⣷⣿⣷⣿⣯⣿⣟⣿⣿⣽⣿⣯⣿⣿⣽⣿⣿⢿⣿
⣿⣿⣿⣿⣿⣿⢿⣟⣿⣽⣾⣿⣟⣿⣟⣯⣿⣿⣿⢿⣋⢉⠪⡮⣳⢯⣟⢼⢯⢗⣟⡮⡪⣳⣝⢕⢎⣾⣿⣷⣿⢿⣻⣽⣿⣽⣿⢿⣻⣯⣿⣿⣽⣿⣯⣿⣿⣻⣿⣿
⣷⣿⣿⣾⣿⣾⣿⣿⣿⣿⡿⣟⣿⡿⣿⣿⣻⡷⣿⣻⣯⣧⡕⢌⢳⣟⡾⣽⣫⢯⡺⣝⢎⢊⢣⢫⣾⣿⣷⢿⣻⣿⡿⣿⣻⣯⣿⣿⣿⣿⣟⣯⣿⣯⣿⣟⣿⣿⣻⣷
⣿⣿⡿⣟⣿⣟⣿⣟⣯⣷⣿⣿⡿⣿⣿⣻⣽⣻⣽⣻⡾⣯⣿⣆⢅⢻⡽⡷⣽⢮⡻⡜⢮⣎⢇⣿⡿⣷⣿⣿⢿⣻⣿⡿⣿⣿⣻⣽⣾⣿⣻⣿⣟⣿⣟⣿⣿⣟⣿⣿
⣿⣷⣿⣿⣿⡿⣿⡿⣿⣟⣯⣷⣿⣟⡾⣷⢿⣟⣿⣽⣿⡽⣾⣻⣦⠑⣟⡯⡯⣗⠽⣽⣵⣿⣿⣽⡿⣿⡷⣿⣿⢿⣯⣿⣿⣯⣿⣿⣻⣽⣿⣻⣿⣻⣿⣟⣯⣿⣿⣷
⣿⣿⣻⣿⣷⣿⣿⢿⣿⢿⣟⣿⣳⡿⣽⣿⣿⣿⢿⣻⣾⣟⣯⢷⣟⣧⢸⣫⣯⣷⡿⣿⢷⡿⣷⢿⣟⣿⣻⣿⣽⣿⢿⣿⣿⣾⣿⣽⡿⣟⣿⣟⣿⣿⣻⣿⡿⣟⣿⣾
⣻⣽⣿⣿⣽⣿⣾⣿⡿⣿⡿⣽⣯⣿⣟⣿⣷⣿⣿⡿⣿⣯⡿⣯⢷⣻⣯⣷⣻⣽⡿⣟⣿⢿⣻⣿⣻⣟⣯⣿⡾⣟⣿⣿⣾⢯⣷⣿⡿⣿⣿⣻⣿⣻⣿⣻⣿⣿⡿⣿
⣿⣿⣻⣽⣿⣷⣿⣷⣿⢿⣻⣿⢷⣿⣿⢿⣻⣿⣽⣿⢿⣿⡽⣟⣯⢿⣺⡽⣎⣿⣟⣿⣻⡿⣯⣿⣯⣿⣻⣷⡿⣟⣿⡾⣟⣿⡿⣽⣿⣿⣽⣿⣻⣿⣻⣿⣯⣷⣿⣿
⣽⣿⡿⣟⣿⣾⣿⣾⢿⣟⣿⣿⣻⣿⣻⣿⣿⣻⣯⣿⢿⣿⣟⣯⣿⡯⣷⣻⡽⡷⣟⣯⣿⣻⣯⣷⡿⣾⣟⣷⡿⣟⣿⣻⣟⣯⣿⣯⣷⡿⣯⣿⣟⣿⣻⣽⣿⣽⣿⣽
⣿⣻⣿⡿⣿⡿⣾⡿⣿⢿⣟⣿⡿⣿⡿⣿⣾⡿⣟⣿⣻⣿⣟⣾⣿⣯⡯⣷⣻⣽⢿⣯⣿⣽⣷⢿⣻⣯⣿⣽⡿⣟⣿⣻⣟⣯⣷⣿⣷⣽⢿⣻⣿⣻⣿⣟⣯⣿⣟⣿
⣻⣿⣻⣿⡿⣿⣿⣻⣿⣿⣿⣿⢿⣿⣿⢿⣷⣿⡿⣿⣻⣾⣿⣽⣿⣷⢿⣽⣳⡯⣿⡷⣟⣷⣿⣻⣿⣽⣷⢿⣻⣿⣻⣟⣿⣯⣿⣾⣷⣿⢿⣻⣿⣻⣽⣿⣻⣿⣻⣿
⢿⣻⣿⣟⣿⣿⣽⡿⣷⡿⣿⣾⣿⣿⣾⣿⣿⣾⣿⣻⣯⣷⣟⣿⣿⣟⣿⣞⣗⣿⢽⣿⣻⣽⡾⣿⢾⣷⢿⣟⣿⣽⣿⣽⡿⣾⢿⣾⢷⣿⢿⣿⣽⢿⣻⣿⣻⣿⣻⣿
⣿⣿⣟⣿⣿⣽⣿⣻⣿⢿⣿⣿⣯⣿⣿⡿⣿⣻⣿⣿⣞⣿⢿⣿⣿⣻⣷⢿⣞⣞⣯⣿⢯⣷⡿⣟⣿⣽⡿⣯⡿⣷⡿⣾⡿⣿⣟⣿⣟⣿⣟⣿⣾⣿⢿⣻⣿⣿⣿⡿
⣿⣯⣿⣿⣽⣿⣽⣿⣻⣿⣿⡿⣿⣾⣿⣿⢿⣿⢿⣻⡿⣯⣿⣿⣿⣻⣿⣻⣽⡾⣽⣾⢿⡷⣿⣻⣯⣷⡿⣟⣿⣯⣿⣟⣿⣯⣿⣯⣿⣯⣿⣯⣷⣿⣿⣿⣷⣿⣿⣿
⣿⣻⣯⣿⣟⣯⣿⣟⣿⣽⣷⣿⣿⣿⣿⣾⣿⣿⣿⡿⣟⣿⣾⣿⣿⣟⣯⡿⣞⣿⣯⣿⣟⣿⣟⣯⣿⡾⣿⣻⡿⣾⡷⣿⣻⣾⣷⡿⣷⣿⣷⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⡿⣿⡿⣿⡿⣿⣻⣿⣻⣿⣿⣿⣿⣽⣿⣿⣿⣿⣿⣿⡿⣿⣽⣾⣿⢿⣟⣿⣺⣷⣿⣯⣷⡿⣯⣷⡿⣟⣿⣻⣟⣿⣿⣟⣿⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⢿⣿⣿⣿⣿⡿⣿⣟⣿⣿⣷⣿⣿⣿⣿⣽⣾⣿⣿⣿⣿⣿⣿⣯⣿⣿⣿⣿⣿⣿⣿⣽⣾⢿⣻⣷⡿⣿⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⢿⣷⣿⣷⣿⣿⣿⢿⣷⣿⣿⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣽⣿⣿⣿⣿⣿⣿⣿⣿⡿⣾⡿⣟⣷⣿⢿⣿⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣾⣿⣿⣿⣽⣿⣷⣿⣿⣿⢿⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣽⣟⡿⣾⣻⣻⣽⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣟⣿⣯⣿⣾⣿⣿⣾⡿⣟⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢇⣯⣻⣺⣞⣷⣿⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⡿⣿⡿⣿⣾⣿⣿⣿⣿⣿⣿⣿⣿⣷⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣽⣽⣾⣾⡿⣿⣿⣿⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
"""

class StyleTransfer(nn.Module):
    def __init__(self):
        super().__init__()

        vgg19 = torchvision.models.vgg19(weights=VGG19_Weights.DEFAULT).features

        for param in vgg19.parameters():
            param.requires_grad = False

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

        out1 = self.e1(input)
        features.append(out1)

        out2 = self.e2(out1)
        features.append(out2)

        out3 = self.e3(out2)
        features.append(out3)

        out4 = self.e4(out3)
        features.append(out4)

        out = self.e5(out4)

        return out, features
    
    def bfa(self, input, concat):
        input = torch.concat([input, self.max1(concat[0])], dim=1)
        input = torch.concat([input, self.max2(concat[1])], dim=1)
        input = torch.concat([input, self.max3(concat[2])], dim=1)
        input = torch.concat([input, self.max4(concat[3])], dim=1)

        return input
    
    def insl(self, content_features, style_features):
        insl_features = []

        insl_features.append(self.wct(self.in1(content_features[0]), self.in1(style_features[0])))
        insl_features.append(self.wct(self.in2(content_features[1]), self.in2(style_features[1])))
        insl_features.append(self.wct(self.in3(content_features[2]), self.in3(style_features[2])))
        insl_features.append(self.wct(self.in4(content_features[3]), self.in4(style_features[3])))
        
        return insl_features

    def forward(self, content, style=None):
        # Training mode, no wct
        if style is None:
            content, content_features = self.encoder(content)

            insl = self.insl(content_features)

            content = self.bfa(content, content_features)

            content = self.d5(content, None)
            content = self.d4(content, self.in4(content_features[3]))
            content = self.d3(content, self.in3(content_features[2]))
            content = self.d2(content, self.in2(content_features[1]))
            content = self.d1(content, self.in1(content_features[0]))

            _, content_features_loss = self.encoder(content)

            return content, content_features, content_features_loss
        
        content, content_features = self.encoder(content)
        style, style_features = self.encoder(style)

        insl = self.insl(content_features, style_features)

        content_wct = self.wct(content, style)

        content_wct = self.bfa(content_wct, content_features)
        style = self.bfa(style, style_features)

        content_wct = self.d5(content_wct, None)
        style = self.d5(style, None)
        content_wct = self.wct(content_wct, style)

        content_wct = self.d4(content_wct, insl[3])
        style = self.d4(style, style_features[3])
        content_wct = self.wct(content_wct, style)

        content_wct = self.d3(content_wct, insl[2])
        style = self.d3(style, style_features[2])
        content_wct = self.wct(content_wct, style)

        content_wct = self.d2(content_wct, insl[1])
        style = self.d2(style, style_features[1])
        content_wct = self.wct(content_wct, style)

        content_wct = self.d1(content_wct, insl[0])

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