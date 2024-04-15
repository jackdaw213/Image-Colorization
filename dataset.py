import torch
import os
import torchvision.transforms as transforms
from cucim.skimage.color import rgb2lab
from PIL import Image
import torch.nn.functional as F

import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn

import utils

class ColorDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, do_transform=False, color_peak=False):
        self.names = os.listdir(image_dir)
        self.image_dir = image_dir
        self.do_transform = do_transform
        self.color_peak = color_peak
        self.transform = transforms.Compose([
            transforms.Resize((248, 248), antialias=True)
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

        if self.color_peak:
            mask = (torch.rand((248, 248)) > 0.95).float()
            black = torch.cat([black, color * mask], dim=0)
        else:
            black = F.pad(black, (0, 0, 0, 0, 1, 1), mode='constant', value=0)

        return black, color, mask
    
    @staticmethod
    @pipeline_def(device_id=0)
    def dali_pipeline(image_dir, color_peak=False):
        images, _ = fn.readers.file(file_root=image_dir, 
                                    files=utils.list_images(image_dir),
                                    random_shuffle=True, 
                                    name="Reader")
        
        images = fn.decoders.image(images, device="mixed", output_type=types.RGB)

        images = fn.resize(images, size=512)

        # By default this function output layout is CHW which will make the program crash with 
        # SIGSEGV (Address boundary error) because rgb2lab needs HWC
        images = fn.crop_mirror_normalize(images, 
                                        dtype=types.FLOAT,
                                        output_layout="HWC",
                                        crop=(256, 256),
                                        crop_pos_x=fn.random.uniform(range=(0, 1)),
                                        crop_pos_y=fn.random.uniform(range=(0, 1)),
                                        mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225])

        images = fn.python_function(images, function=rgb2lab)

        images = fn.transpose(images, perm=[2, 0, 1])
        
        color = images[1:, :, :]
        black = fn.expand_dims(images[0, :, :], axes=0)

        # If color_peak is true then we will feed the model with color information
        # False if we want only the chrominance information
        if color_peak:
            # In the paper the author feeds the model a peek of the color of the image
            # to force the network to complete the ab information. How much information
            # about the color channel is given to the network is unknown as it does not 
            # mention by the author. So I will give the network only 5% of the information

            # TODO: Change torch method to DALI one because mask gets copied back to RAM 
            # everytime we return it even with device="cuda". I do not know if this is a
            # bug or not
            mask = (torch.rand((256, 256), device="cuda") > 0.95).float()
            black = fn.cat(black, color * mask, axis=0)
        else:
            # Input is gray scale image with 1 channel, resnet needs 3 so we need
            # to pad the image with extra 2 channels of 0
            black = F.pad(black, (0, 0, 0, 0, 1, 1), mode='constant', value=0)

        return black, color, mask
    
# So the dataset I'm using for style representation is this one https://www.kaggle.com/c/painter-by-numbers/data
# There are some problems with it, the first thing is that there are some gif images with jpg extention which 
# DALI does not like. Second, there are some images with monsterous resolution hiding inside the dataset, the 
# largest one I found was 30000x29605 - 220MB (⌐■_■) ( •_•)>⌐■-■. These images makes the program crashed with 
# OOM error which confused me a bit tbh, thought it was because of a bug in my dali_pipeline
class StyleDataset(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass
    
    @staticmethod
    @pipeline_def(device_id=0)
    def dali_pipeline(content_dir, style_dir):
        content_images, _ = fn.readers.file(file_root=content_dir, 
                                            files=utils.list_images(content_dir),
                                            random_shuffle=True, 
                                            name="Reader")
        
        style_images, _ = fn.readers.file(file_root=style_dir, 
                                            files=utils.list_images(style_dir),
                                            random_shuffle=True)
        
        content_images = fn.decoders.image(content_images, device="mixed", output_type=types.RGB)
        style_images = fn.decoders.image(style_images, device="mixed", output_type=types.RGB)

        content_images = fn.resize(content_images, size=512, dtype=types.FLOAT)
        style_images = fn.resize(style_images, size=512, dtype=types.FLOAT)

        content_images = fn.crop_mirror_normalize(content_images, 
                                                dtype=types.FLOAT,
                                                crop=(256, 256),
                                                crop_pos_x=fn.random.uniform(range=(0, 1)),
                                                crop_pos_y=fn.random.uniform(range=(0, 1)),
                                                mean=[0.485, 0.456, 0.406], 
                                                std=[0.229, 0.224, 0.225])
        style_images = fn.crop_mirror_normalize(style_images, 
                                                dtype=types.FLOAT,
                                                crop=(256, 256),
                                                crop_pos_x=fn.random.uniform(range=(0, 1)),
                                                crop_pos_y=fn.random.uniform(range=(0, 1)),
                                                mean=[0.485, 0.456, 0.406], 
                                                std=[0.229, 0.224, 0.225])

        return content_images, style_images