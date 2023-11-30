import torch
import os
import torchvision.transforms.functional as f
import torchvision.transforms as transforms
from skimage.color import rgb2lab
from torchvision.io import read_image
from PIL import Image

class ColorDataset(torch.utils.data.Dataset):
    def __init__(self, black_dir, color_dir):
        self.names = os.listdir(color_dir)
        self.black_dir = black_dir
        self.color_dir = color_dir

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        black_path = os.path.join(self.black_dir, self.names[index])
        color_path = os.path.join(self.color_dir, self.names[index])

        color_img = torch.from_numpy(rgb2lab(read_image(color_path).permute(1, 2, 0))).permute(2, 0, 1).float()
        color = color_img[1:, :, :]
        black = color_img[0, :, :].unsqueeze(0)

        return black, color
