import torch.nn as nn
import auto_parts as ap
import torch.nn.functional as F

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

        self.hc = ap.HyperConnections(64)

        self.out = ap.OutConv(64, 2)

    def forward(self, x):
        r1_e_f, r1_e_d = self.e1(x)
        r2_e_f, r2_e_d = self.e2(r1_e_d)
        r3_e_f, r3_e_d = self.e3(r2_e_d)
        r4_e_f, r4_e_d = self.e4(r3_e_d)

        ls1 = self.ls(r4_e_d)

        r4_d = self.d4(ls1, r4_e_f)
        r3_d = self.d3(r4_d, r3_e_f)
        r2_d = self.d2(r3_d, r2_e_f)
        r1_d = self.d1(r2_d, r1_e_f)

        hc1 = self.hc(r1_d, r1_e_f, r2_e_f, r2_d)

        return self.out(hc1)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')