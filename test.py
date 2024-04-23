import torch
import torchvision.transforms.functional as F
import torch.nn.functional as f
import matplotlib.pyplot as plt
import argparse

from skimage.color import rgb2lab
from skimage.color import rgb2gray
from PIL import Image

import model
import utils

GRAY_IMAGE = "model/cus_img/gray2.jpg"
REF_IMAGE = "model/cus_img/ref2.jpg"

parser = argparse.ArgumentParser(description='Image colorization using UNet')

parser.add_argument('-g', '--gray_image', type=str,
                    default=GRAY_IMAGE,
                    help='Path to the gray scale image')
parser.add_argument('-r', '--ref_image', type=str,
                    default=REF_IMAGE,
                    help='Path to the reference image')

args = parser.parse_args()

style = model.StyleTransfer()
style.load_state_dict(torch.load("model/style", 
map_location=torch.device('cpu'))["model"])
style.eval()

color = model.UNetResEncoder()
color.load_state_dict(torch.load("model/train.state", 
map_location=torch.device('cpu'))["model"])
color.eval()

gray_rgb = Image.open(args.gray_image).convert("RGB")
gray_l, gray_ab = utils.rgb_to_l_ab(gray_rgb)
gray_l_original = gray_l
gray_l = f.pad(gray_l, (0, 0, 0, 0, 1, 1), mode='constant', value=0).unsqueeze(dim=0).float()

with torch.no_grad():
    color_ab = color(gray_l)

color_ab = color_ab.squeeze()
auto_gray_rgb = utils.l_ab_to_rgb(gray_l_original, color_ab)
plt.imshow(auto_gray_rgb.permute(1, 2, 0))
plt.show()
auto_gray_rgb = utils.norm(auto_gray_rgb).unsqueeze(dim=0).float()

ref_rgb = Image.open(args.ref_image).convert("RGB")
ref_rgb = F.to_tensor(ref_rgb)
ref_rgb = utils.norm(ref_rgb).unsqueeze(dim=0).float()

with torch.no_grad():
    style_rgb = style(auto_gray_rgb, ref_rgb)

style_rgb = utils.denorm(style_rgb.squeeze())
plt.imshow(style_rgb.permute(1, 2, 0))
plt.show()

style_l, style_ab = utils.rgb_to_l_ab(style_rgb)
gray_lab = torch.cat((gray_l_original, style_ab), dim=0)
gray_lab = gray_lab.unsqueeze(dim=0).float()

with torch.no_grad():
    color_ab_final = color(gray_lab)

color_ab_final = color_ab_final.squeeze()
result = utils.l_ab_to_rgb(gray_l_original, color_ab_final)
plt.imshow(result.permute(1, 2, 0))
plt.show()
