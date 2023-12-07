import torch
import os
import torchvision.transforms.functional as f
import torchvision.transforms as transforms
from skimage.color import rgb2lab
from torchvision.io import read_image
from PIL import Image

class ColorDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, do_transform=False):
        self.names = os.listdir(image_dir)
        self.image_dir = image_dir
        self.do_transform = do_transform
        self.transform = transforms.Compose([
            transforms.Resize((320, 240), antialias=True)
        ])
    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.names[index])

        image = rgb2lab(Image.open(image_path).convert("RGB"))
        image = torch.from_numpy(image).permute(2,0,1).float()
        if self.do_transform:
            image = self.transform(image)

        color = image[1:, :, :]
        black = image[0, :, :].unsqueeze(0)
        
        return black, color
