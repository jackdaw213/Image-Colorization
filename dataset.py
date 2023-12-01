import torch
import os
import torchvision.transforms.functional as f
import torchvision.transforms as transforms
from skimage.color import rgb2lab
from torchvision.io import read_image
from PIL import Image

class ColorDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir):
        self.names = os.listdir(image_dir)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.names[index])

        color_img = torch.from_numpy(rgb2lab(read_image(image_path).permute(1, 2, 0))).permute(2, 0, 1).float()
        color = color_img[1:, :, :]
        black = color_img[0, :, :].unsqueeze(0)

        return black, color
