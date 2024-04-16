import os
import shutil
import sys
import random
import filetype

import torch
import torchmetrics
from torchvision.io import read_image

import matplotlib.pyplot as plt
from skimage.color import lab2rgb
from skimage.color import rgb2lab

import tqdm as tq
import numpy as np
import torch.nn as nn
from PIL import Image
import PIL
from PIL import ImageFile
import torchvision.transforms.functional as F

def image_grid(**kwargs):
    col_names = list(kwargs.keys())
    num_rows = len(kwargs[col_names[0]])
    num_cols = len(col_names)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 9))

    for row in range(num_rows):
        for col in range(num_cols):
            ax = axs[row, col]
            if row == 0:
                ax.set_title(col_names[col])
            ax.imshow(kwargs[col_names[col]][row])
            ax.axis('off')
            
    # Adjust layout
    plt.tight_layout()
    plt.show()

def l_ab_to_rgb(l, ab):
    img = torch.cat([l, ab], 0)
    img = lab2rgb(img.permute(1, 2, 0))
    return img

def rgb_to_l_ab(rgb):
    img = torch.from_numpy(rgb2lab(rgb)).permute(2, 0, 1)
    ab = img[1:, :, :]
    l = img[0, :, :].unsqueeze(0) # Add the channels dimension 
    return l, ab

def concat_l_and_to_rgb(l, ab_shape):
    l = torch.cat([l, torch.zeros(ab_shape)], 0)
    return lab2rgb(l.permute(1, 2, 0))

def test_learnability(model, learning_rate, image_path, n_epochs):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    l, ab = rgb_to_l_ab(image_path)
    ab = ab.unsqueeze(0).float().cuda()
    l = l.unsqueeze(0).float().cuda()

    epoch_loss = torchmetrics.MeanMetric().to(device)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.SmoothL1Loss().cuda()

    train_list = []

    model.train()
    for epoch in tq.tqdm(range(n_epochs), total=n_epochs, desc='Epochs'):
    
        output = model(l)
        loss = loss_func(output, ab)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss(loss)
        train_list.append(epoch_loss.compute().cpu())

    model.eval()
    with torch.no_grad():
        out = model(l)
        l = l.cpu().squeeze(0)
        out = out.cpu().squeeze(0)
        ab = ab.cpu().squeeze(0)
        result = l_ab_to_rgb(l, out)
        ground = l_ab_to_rgb(l, ab)

        compare_img((concat_l_and_to_rgb(l, ab.shape), result, ground))

    plt.plot(train_list, label='train_loss')
    plt.legend()
    plt.show()

def test_trained_model(model, test_images_path, num_samples=8):
    images = os.listdir(test_images_path)
    selected_images = np.random.choice(images, num_samples, replace=False)
    image_list = []
    for image_name in selected_images:   
        image_path = os.path.join(test_images_path, image_name) 
        l, ab = rgb_to_l_ab(image_path)
        mask = (torch.rand((ab.shape[1], ab.shape[2])) > 0.95).float()
        inp = torch.cat([l, ab * mask], dim=0)
        ab = ab.unsqueeze(0).float()
        inp = inp.unsqueeze(0).float()
        model.eval()
        with torch.no_grad():
            out = model(inp)
            out = out.squeeze(0)
            ab = ab.squeeze(0)
            image_list.append((concat_l_and_to_rgb(l, ab.shape), l_ab_to_rgb(l, out), l_ab_to_rgb(l, ab)))
    compare_img_table(image_list, num_samples)              

def plot_loss_history(state_file):
    state = torch.load(state_file)
    plt.plot(state["train_list"], label='train_loss')
    plt.plot(state["val_list"],label='val_loss')
    plt.legend()
    plt.show()

def save_train_state(model, optimizer, epoch, path):
    # This one is for resuming training
    torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch
    }, path)

    path = path + ".epoch" + str(epoch + 1)
    torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch
    }, path)

def load_train_state(path):
    try:
        state = torch.load(path)
        return state["model"], state["optimizer"], state["epoch"]
    except Exception as e:
        print(e)
        print("Error encounted when trying to load train state, trying backup")
        try:
            torch.load(path + ".bak")
            return state["model"], state["optimizer"], state["epoch"]
        except:
            print(e)
            sys.exit("Backup train state failed, existing")

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

def split_images(input_folder, output_folder, train_ratio=0.8, val_ratio=0.15, test_ratio=0.05):
    if train_ratio + val_ratio + test_ratio != 1:
        raise ValueError("Ratio sum must be 1")
    # Create output folders if they don't exist
    train_folder = os.path.join(output_folder, 'train')
    val_folder = os.path.join(output_folder, 'val')
    test_folder = os.path.join(output_folder, 'test')

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Get a list of all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

    # Shuffle the list of image files
    random.shuffle(image_files)

    # Calculate the number of images for each set
    total_images = len(image_files)
    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)
    test_count = total_images - train_count - val_count

    # Copy images to the corresponding folders
    for i, image_file in enumerate(image_files):
        source_path = os.path.join(input_folder, image_file)

        if i < train_count:
            destination_folder = train_folder
        elif i < train_count + val_count:
            destination_folder = val_folder
        else:
            destination_folder = test_folder

        destination_path = os.path.join(destination_folder, image_file)
        shutil.copy(source_path, destination_path)

    print(f"Splitting complete. Train: {train_count}, Val: {val_count}, Test: {test_count}")

def create_small_dataset(original_folder, target_folder, split_ratio=0.1):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    all_files = os.listdir(original_folder)

    num_files = len(all_files)
    num_files = int(num_files * split_ratio)

    small_files = random.sample(all_files, num_files)

    for file_name in small_files:
        source_path = os.path.join(original_folder, file_name)
        target_path = os.path.join(target_folder, file_name)
        shutil.copyfile(source_path, target_path)

def mean_std(input):
    mean = torch.mean(input, dim=(2, 3), keepdims=True)
    # For some reasons the torch.std() uses Bessel’s correction by default so unbiased=False 
    # is used to make standard deviation "standard". And also keepdims to makes this function
    # correctly
    # Add 1e-6 to avoid dividing by zero which cause AdaIN's output to contain NaN
    std = torch.std(input, dim=(2, 3), unbiased=False, keepdims=True) + 1e-6
    return mean, std

def pad_fetures(up, con_channels):
    # We need to pad the features with 0 when we concatenating upscaled 
    # features that were previously downscaled from odd dimension features
    # For example: 25 -> down -> 12 -> up -> 24 -> pad -> 25
    diffY = con_channels.size()[2] - up.size()[2]
    diffX = con_channels.size()[3] - up.size()[3]
    up = torch.nn.functional.pad(up, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
    return up

# Instead of creating a labels file, we can just pass a list of files to the 
# decoder via files argument. And it does not take too much time either (2s 
# from my testing)
def list_images(folder_path):
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

# https://stackoverflow.com/questions/33548956/detect-avoid-premature-end-of-jpeg-in-cv2-python
# This can remove MOST corrupted images from my testing but good enough I guess 
# Also this method is much faster than Image.open(path).convert("RGB") with try-except
def remove_corrupted_jpeg(folder_path):
    for filename in os.listdir(folder_path):
        path = f"{folder_path}/{filename}"
        with open(path, 'rb') as f:
            check_chars = f.read()[-2:]
            if check_chars != b'\xff\xd9':
                os.remove(path)