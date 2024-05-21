import os
import sys
import random
import torch

import matplotlib.pyplot as plt
from skimage.color import lab2rgb
from skimage.color import rgb2lab

import numpy as np
from PIL import Image
import PIL
import torchvision.transforms.functional as F

def image_grid(**kwargs):
    col_names = list(kwargs.keys())
    num_rows = len(kwargs[col_names[0]])
    num_cols = len(col_names)
        
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 9))

    if num_rows != 1 and num_cols != 1:
        for row in range(num_rows):
            for col in range(num_cols):
                ax = axs[row, col]
                if row == 0:
                    ax.set_title(col_names[col])
                ax.imshow(kwargs[col_names[col]][row])
                ax.axis('off')
    elif num_rows != 1 and num_cols == 1:
        for row in range(num_rows):
            ax = axs[row]
            if row == 0:
                ax.set_title(col_names[0])
            ax.imshow(kwargs[col_names[0]][row])
            ax.axis('off')
            
    elif num_rows == 1 and num_cols != 1:
        for col in range(num_cols):
            ax = axs[col]
            ax.set_title(col_names[col])
            ax.imshow(kwargs[col_names[col]][0])
            ax.axis('off')
    else:
        axs.set_title(col_names[0])
        axs.imshow(kwargs[col_names[0]][0])
        axs.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

def l_ab_to_rgb(l, ab):
    """
    Input 2 CHW tensors -> Output 1 CHW tensor
    """
    img = torch.cat([l * 100 + 50, ab  * 110], 0)
    img = lab2rgb(img.permute(1, 2, 0))
    img = torch.from_numpy(img)
    return img.permute(2, 0, 1)

def rgb_to_l_ab(rgb):
    """
    Input 1 CHW tensors or PIL image -> Output 2 CHW tensor
    """
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.permute(1, 2, 0)
    img = torch.from_numpy(rgb2lab(rgb)).permute(2, 0, 1)
    ab = img[1:, :, :]
    l = img[0, :, :].unsqueeze(0) # Add the channels dimension 
    return (l - 50) / 100, ab / 110

def concat_l_and_to_rgb(l, ab_shape):
    l = torch.cat([l, torch.zeros(ab_shape)], 0)
    return lab2rgb(l.permute(1, 2, 0))

def test_color_model(model, test_images_path, color_peak=True, num_samples=8):
    images = os.listdir(test_images_path)
    images = np.random.choice(images, num_samples, replace=False)

    input = []
    output = []
    ground_truth = []

    for image_name in images:   
        image_path = os.path.join(test_images_path, image_name) 
        pil_img = Image.open(image_path).convert("RGB")

        ground_truth.append(pil_img)

        pil_img = F.to_tensor(pil_img)
        l, ab = rgb_to_l_ab(pil_img)

        if color_peak:
            mask = (torch.rand((ab.shape[1], ab.shape[2])) > 0.95).float()
            inp = torch.cat([l, ab * mask], dim=0)
            input.append(lab2rgb(torch.cat((l*100+50, ab * mask), dim=0).permute(1, 2, 0)))
        else:
            inp = torch.cat((l, torch.zeros(2, ab.shape[1], ab.shape[2])), dim=0)
            input.append(lab2rgb(torch.cat((l*100+50, torch.zeros(2, ab.shape[1], ab.shape[2])), dim=0).permute(1, 2, 0)))

        inp = inp.unsqueeze(0).float()

        model.eval()
        with torch.no_grad():
            out = model(inp)
        out = out.squeeze(0)

        output.append(l_ab_to_rgb(l, out).permute(1, 2, 0))

    image_grid(Input=input, Output=output, Ground_Truth=ground_truth)             

def save_train_state(model, optimizer, scaler, epoch, path):
    # This one is for resuming training
    torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scaler': scaler.state_dict(),
    'epoch': epoch
    }, path)

    path = path + ".epoch" + str(epoch + 1)
    torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scaler': scaler.state_dict(),
    'epoch': epoch
    }, path)

def load_train_state(path):
    try:
        state = torch.load(path)
        return state["model"], state["optimizer"], state["scaler"], state["epoch"]
    except Exception as e:
        print(e)
        sys.exit("Loading train state failed, existing")

def pad_fetures(up, con_channels):
    """
    We need to pad the features with 0 when we concatenating upscaled 
    features that were previously downscaled from odd dimension features
    For example: 25 -> down -> 12 -> up -> 24 -> pad -> 25
    """
    diffY = con_channels.size()[2] - up.size()[2]
    diffX = con_channels.size()[3] - up.size()[3]
    up = torch.nn.functional.pad(up, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
    return up

def list_images(folder_path):
    """
    Instead of creating a labels file, we can just pass a list of files to the 
    decoder via files argument. And it does not take too much time either (2s 
    from my testing)
    """
    temp = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            temp.append(filename)
    return temp

def delete_grayscale_images(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = f"{folder_path}/{filename}"
            image = Image.open(path).convert("RGB")

            tensor = F.pil_to_tensor(image)

            mean_r = torch.mean(tensor[0].float())
            mean_g = torch.mean(tensor[1].float())
            mean_b = torch.mean(tensor[2].float())

            if mean_r == mean_g == mean_b:
                os.remove(path)
