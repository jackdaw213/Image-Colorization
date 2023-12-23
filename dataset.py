import torch
import os
import torchvision.transforms.functional as f
import torchvision.transforms as transforms
from cucim.skimage.color import rgb2lab
from torchvision.io import read_image
from PIL import Image

import cv2
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn

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

    @staticmethod
    def smt(image):
        print(image.shape)
        return image
    
    @staticmethod
    @pipeline_def(device_id=0)
    def dali_pipeline(image_dir, do_transform=True):
        images, _ = fn.readers.file(file_root=image_dir, 
                                         file_list=image_dir+"/labels.txt",
                                         random_shuffle=True, 
                                         name="Reader")
        
        images = fn.decoders.image(images, device="mixed", output_type=types.RGB)
        images = fn.python_function(images, function=rgb2lab)

        if do_transform:
            images = fn.resize(images, size=(320, 240))

        images = fn.transpose(images, perm=[2, 0, 1])
        
        #assd = fn.python_function(images, function=ColorDataset.smt)
        
        color = images[1:, :, :]
        black = fn.expand_dims(images[0, :, :], axes=0)

        return black, color