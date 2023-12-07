import torch.nn as nn
import auto_parts as ap
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.e1 = ap.EncoderBlock(1, 64)
        self.e2 = ap.EncoderBlock(64, 128)
        self.e3 = ap.EncoderBlock(128, 256)

        self.ls = ap.LatentSpace(256, 512)

        self.d3 = ap.DecoderBlock(512, 256)
        self.d2 = ap.DecoderBlock(256, 128)
        self.d1 = ap.DecoderBlock(128, 64)

        self.out = ap.OutConv(64, 2)

    def forward(self, x):
        r1_e_f, r1_e_d = self.e1(x)
        r2_e_f, r2_e_d = self.e2(r1_e_d)
        r3_e_f, r3_e_d = self.e3(r2_e_d)

        ls1 = self.ls(r3_e_d)

        r3_d = self.d3(ls1, r3_e_f)
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

        self.d3 = ap.CDecoderBlock(512, 256)
        self.d2 = ap.CDecoderBlock(256, 128)
        self.d1 = ap.CDecoderBlock(128, 64)

        self.out = ap.COutConv(64, 2)

    def forward(self, x):
        r1_e_f, r1_e_d = self.e1(x)
        r2_e_f, r2_e_d = self.e2(r1_e_d)
        r3_e_f, r3_e_d = self.e3(r2_e_d)

        ec1 = self.ec1(r3_e_d)

        r3_d = self.d3(ec1, r3_e_f)
        r2_d = self.d2(r3_d, r2_e_f)
        r1_d = self.d1(r2_d, r1_e_f)

        return self.out(r1_d)