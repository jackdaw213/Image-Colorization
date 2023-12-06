import torch.nn as nn
import auto_parts as ap
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.e1 = ap.EncoderBlock(1, 32)
        self.e2 = ap.EncoderBlock(32, 64)
        self.e3 = ap.EncoderBlock(64, 128)
        self.e4 = ap.EncoderBlock(128, 256)
        self.e5 = ap.EncoderBlock(256, 512)

        self.ls = ap.LatentSpace(512, 1024)

        self.d5 = ap.DecoderBlock(1024, 512)
        self.d4 = ap.DecoderBlock(512, 256)
        self.d3 = ap.DecoderBlock(256, 128)
        self.d2 = ap.DecoderBlock(128, 64)
        self.d1 = ap.DecoderBlock(64, 32)

        self.out = ap.OutConv(32, 2)

    def forward(self, x):
        r1_e_f, r1_e_d = self.e1(x)
        r2_e_f, r2_e_d = self.e2(r1_e_d)
        r3_e_f, r3_e_d = self.e3(r2_e_d)
        r4_e_f, r4_e_d = self.e4(r3_e_d)
        r5_e_f, r5_e_d = self.e5(r4_e_d)

        ls1 = self.ls(r5_e_d)

        r5_d = self.d5(ls1, r5_e_f)
        r4_d = self.d4(r5_d, r4_e_f)
        r3_d = self.d3(r4_d, r3_e_f)
        r2_d = self.d2(r3_d, r2_e_f)
        r1_d = self.d1(r2_d, r1_e_f)

        return self.out(r1_d)
    
#https://www.mdpi.com/2073-8994/14/11/2295    
class CUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.e1 = ap.CEncoderBlock(1, 64)
        self.e2 = ap.CEncoderBlock(64, 128)
        self.e3 = ap.CEncoderBlock(128, 256)

        self.ec1 = ap.ExtendedConvolution(256, 512)
        self.ec2 = ap.ExtendedConvolution(512, 512)

        self.d3 = ap.CDecoderBlock(512, 256)
        self.d2 = ap.CDecoderBlock(256, 128)
        self.d1 = ap.CDecoderBlock(128, 64)

        self.out = ap.OutConv(64, 2)

    def forward(self, x):
        r1_e_f, r1_e_d = self.e1(x)
        r2_e_f, r2_e_d = self.e2(r1_e_d)
        r3_e_f, r3_e_d = self.e3(r2_e_d)

        ec1 = self.ec1(r3_e_d)
        ec2 = self.ec2(ec1)

        r3_d = self.d3(ec2, r3_e_f)
        r2_d = self.d2(r3_d, r2_e_f)
        r1_d = self.d1(r2_d, r1_e_f)

        return self.out(r1_d)