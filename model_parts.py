import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

# Has anyone ever felt so embarrassed by their commit that they git reset --hard HEAD~1
# a few times to erase their sins from history, or is it just me ¯\_(ツ)_/¯

class ColorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model_out, ground_truth, mask, weight_coefficient=10):
        mask = 1 + weight_coefficient * mask
        model_out = mask * model_out
        ground_truth = mask * ground_truth
        return F.huber_loss(model_out, ground_truth)


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
        # Features is for concatenating with the decoder block features
        # Down is for the next encoder block
        return features, down 

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, input_channels=False, kernel_size=3, padding=1):
        super().__init__()

        self.up_scale = nn.Upsample(scale_factor=2, mode='nearest')

        if not input_channels:
            # We double the number of channels because we are going to concatenating an equal 
            # number of channels from the encoder block
            in_channels = in_channels * 2
        else:
            # Or we concatenating with the input image
            in_channels = in_channels + input_channels

        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )

    def forward(self, inp, con_channels):
        up = self.up_scale(inp)
        up = utils.pad_fetures(up, con_channels)
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

        """
        So there is a bug that has been plaguing this project from the start: the model can NOT 
        output green (sometimes blue color). At first, I thought it was because my model was 
        trash and I didn't have enough data for the model. So I added ResNet34 as an encoder 
        and used ImageNet as the dataset, the same problem still exists. Until yesterday, I 
        decided to print out the ab channels of the input image and the model output to compare. 
        And I noticed that the ab channels from the input have negative values, while the model 
        output has lots of zeros. A quick Google search and... ab channels have the range of 
        [−128, 127] where NEGATIVE values represent GREEN AND BLUE. And guess what was the final 
        layer of OutConv? A goddamn ReLU().
        I removed it, and now the model can output green and blue. :D
        
        PS: I was actually considering using LeakyReLU() but then realized that LeakyReLU with
        slope of 1 is exactly not having an activation function at all
        """

        """
        Another story for future me (or anyone reading this). When I removed the ReLU() layer, I 
        was wayyyy to excited and did not notice the BatchNorm2d() layer chilling here. This cause 
        the network to output images with horizontal red lines all over the place. This is quite 
        an odd case because the lines became more visible after each epoch. At first I thought it 
        might be due to the TransConv layer causing checkerboard effect [1*] (was not the case). 
        And tried to replace it with NN Upsample and redesigned the model class a bit which cut the 
        training time by around 30 minutes with similar and sometime slightly better quality output 
        (probably just a placebo effect).

        [1*]: https://distill.pub/2016/deconv-checkerboard/
        """
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
    def forward(self, inp):
        return self.out(inp)
