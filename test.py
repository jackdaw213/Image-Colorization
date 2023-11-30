import time
import torch
import torch.nn as nn
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms.functional as f
import torch.nn.functional as F

from skimage.color import rgb2lab
from PIL import Image
from torch.utils.data import DataLoader
import cv2

import auto_parts as ap
import model
import dataset
import utils

model = model.CAE()
utils.test_learnability(model, "img/in/0bbRiP.jpg", 10000)


# model = model.CAE()
# model.load_state_dict(torch.load("model/test"))
# model.eval()
