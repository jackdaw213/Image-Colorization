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

    if num_rows != 1:
        for row in range(num_rows):
            for col in range(num_cols):
                ax = axs[row, col]
                if row == 0:
                    ax.set_title(col_names[col])
                ax.imshow(kwargs[col_names[col]][row])
                ax.axis('off')
    else:
        for col in range(num_cols):
            ax = axs[col]
            ax.set_title(col_names[col])
            ax.imshow(kwargs[col_names[col]][0])
            ax.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

def l_ab_to_rgb(l, ab):
    img = torch.cat([l, ab], 0)
    img = lab2rgb(img.permute(1, 2, 0))
    img = torch.from_numpy(img)
    return img.permute(2, 0, 1)

def rgb_to_l_ab(rgb):
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.permute(1, 2, 0)
    img = torch.from_numpy(rgb2lab(rgb)).permute(2, 0, 1)
    ab = img[1:, :, :]
    l = img[0, :, :].unsqueeze(0) # Add the channels dimension 
    return l, ab

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
            inp = torch.cat([l/100, ab * mask], dim=0)
            input.append(lab2rgb(torch.cat((l, torch.zeros(2, ab.shape[1], ab.shape[2])), dim=0).permute(1, 2, 0)))
        else:
            inp = torch.cat((l/100, torch.zeros(2, ab.shape[1], ab.shape[2])), dim=0)
            input.append(lab2rgb(torch.cat((l, torch.zeros(2, ab.shape[1], ab.shape[2])), dim=0).permute(1, 2, 0)))

        ab = ab.unsqueeze(0).float()
        inp = inp.unsqueeze(0).float()

        model.eval()
        with torch.no_grad():
            out = model(inp)
        out = out.squeeze(0)
        ab = ab.squeeze(0)
        output.append(l_ab_to_rgb(l, out).permute(1, 2, 0))

    image_grid(Input=input, Output=output, Ground_Truth=ground_truth)     
    
def test_style_model(model, con_images_path, sty_images_path, num_samples=8):
    cons = os.listdir(con_images_path)
    cons = np.random.choice(cons, num_samples, replace=False)

    stys = os.listdir(sty_images_path)
    stys = np.random.choice(stys, num_samples, replace=False)

    con_images = []
    sty_images = []
    output = []

    for con, sty in zip(cons, stys):   
        con = os.path.join(con_images_path, con) 
        con = Image.open(con).convert("RGB")
        con_images.append(con)
        con = norm_pil(con)

        sty = os.path.join(sty_images_path, sty) 
        sty = Image.open(sty).convert("RGB")
        sty_images.append(sty)
        sty = norm_pil(sty)

        con = con.unsqueeze(0).float()
        sty = sty.unsqueeze(0).float()

        model.eval()
        with torch.no_grad():
            out = model(con, sty)
        out = denorm(out.squeeze())
        output.append(F.to_pil_image(out))

    image_grid(Content=con_images, Style=sty_images, Output=output)            

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

def norm(tensor):
    return F.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def norm_pil(image):
    ten = F.to_tensor(image)
    return norm(ten)

def denorm(tensor):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res

def whiten_and_color(cF,sF):
    cFSize = cF.size()
    c_mean = torch.mean(cF,1) # c x (h x w)
    c_mean = c_mean.unsqueeze(1).expand_as(cF)
    cF = cF - c_mean

    contentConv = torch.mm(cF,cF.t()).div(cFSize[1]-1) + torch.eye(cFSize[0]).double()
    c_u,c_e,c_v = torch.svd(contentConv,some=False)

    k_c = cFSize[0]
    for i in range(cFSize[0]):
        if c_e[i] < 0.00001:
            k_c = i
            break

    sFSize = sF.size()
    s_mean = torch.mean(sF,1)
    sF = sF - s_mean.unsqueeze(1).expand_as(sF)
    styleConv = torch.mm(sF,sF.t()).div(sFSize[1]-1)
    s_u,s_e,s_v = torch.svd(styleConv,some=False)

    k_s = sFSize[0]
    for i in range(sFSize[0]):
        if s_e[i] < 0.00001:
            k_s = i
            break

    c_d = (c_e[0:k_c]).pow(-0.5)
    step1 = torch.mm(c_v[:,0:k_c],torch.diag(c_d))
    step2 = torch.mm(step1,(c_v[:,0:k_c].t()))
    whiten_cF = torch.mm(step2,cF)

    s_d = (s_e[0:k_s]).pow(0.5)
    targetFeature = torch.mm(torch.mm(torch.mm(s_v[:,0:k_s],torch.diag(s_d)),(s_v[:,0:k_s].t())),whiten_cF)
    targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
    return targetFeature

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

def resize_large_images(folder_path):
    # Temporary raise the maximum image pixels to avoid PIL.Image.DecompressionBombError
    PIL.Image.MAX_IMAGE_PIXELS = 933120000

    max_res = 3840 * 2160
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = f"{folder_path}/{filename}"
            image = Image.open(path).convert("RGB")
            tensor = F.pil_to_tensor(image)

            current_res = tensor.shape[1] * tensor.shape[2]

            if current_res > max_res:
                scale = (max_res / current_res) ** 0.5
                tensor = F.resize(tensor, (int(tensor.shape[1] * scale), int(tensor.shape[2] * scale)))
                image = F.to_pil_image(tensor)
                image.save(os.path.join(folder_path, filename))

def remove_corrupted_jpeg(folder_path):
    """
    https://stackoverflow.com/questions/33548956/detect-avoid-premature-end-of-jpeg-in-cv2-python
    This can remove MOST corrupted images from my testing but good enough I guess 
    Also this method is much faster than Image.open(path).convert("RGB") with try-except
    """
    for filename in os.listdir(folder_path):
        path = f"{folder_path}/{filename}"
        with open(path, 'rb') as f:
            check_chars = f.read()[-2:]
            if check_chars != b'\xff\xd9':
                os.remove(path)
