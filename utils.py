import os
import shutil
import sys
import random
import torch
import torch.nn as nn
import torchmetrics
from torchvision.io import read_image
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
from skimage.color import rgb2lab
import tqdm as tq
import numpy as np
import auto_parts

def compare_img(img, size=(20, 6)):
    inp, out, ground = img

    plt.figure(figsize=size)

    plt.subplot(1, 3, 1)
    plt.imshow(inp)
    plt.title("Input")

    plt.subplot(1, 3, 2)
    plt.imshow(out)
    plt.title("Output")

    plt.subplot(1, 3, 3)
    plt.imshow(ground)
    plt.title("Ground")

    plt.show()

def compare_img_table(image_list, num_samples, size=(20, 20)):
    plt.figure(figsize=size)
    plt.rcParams['figure.dpi'] = 120
    for i in range(len(image_list)):
        inp, out, ground = image_list[i]

        plt.subplot(num_samples, 3, i * 2 + 1 + i)
        plt.axis('off') 
        plt.imshow(inp)

        plt.subplot(num_samples, 3, i * 2 + 2 + i)
        plt.axis('off') 
        plt.imshow(out)

        plt.subplot(num_samples, 3, i * 2 + 3 + i)
        plt.axis('off') 
        plt.imshow(ground)

    plt.show()

def l_ab_to_rgb(l, ab):
    img = torch.cat([l, ab], 0)
    img = lab2rgb(img.permute(1, 2, 0))
    return img

def rgb_to_l_ab(image_path):
    img = torch.from_numpy(rgb2lab(read_image(image_path).permute(1, 2, 0))).permute(2, 0, 1)
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
    loss_func = auto_parts.SmoothL1Loss().cuda()

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
        ab = ab.unsqueeze(0).float()
        l = l.unsqueeze(0).float()
        model.eval()
        with torch.no_grad():
            out = model(l)
            l = l.squeeze(0)
            out = out.squeeze(0)
            ab = ab.squeeze(0)
            image_list.append((concat_l_and_to_rgb(l, ab.shape), l_ab_to_rgb(l, out), l_ab_to_rgb(l, ab)))
    compare_img_table(image_list, num_samples)              
            
def save_train_state(model, optimizer, train_list, val_list, epoch, path, checkpoint=False):
    path_bak = path + ".bak"

    if os.path.isfile(path_bak):
        os.remove(path_bak)

    if os.path.isfile(path): 
        shutil.copy2(path, path_bak)

    if checkpoint:
        path = path + ".epoch" + str(epoch + 1)

    torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'train_list': train_list,
    'val_list': val_list,
    'epoch': epoch
    }, path)

def load_train_state(path, model_only=False):
    try:
        state = torch.load(path)
        if not model_only:
            return state["model"], state["optimizer"], state["train_list"], state["val_list"], state["epoch"]
        else:
            return state["model"]
    except Exception as e:
        print(e)
        print("Error encounted when trying to load train state, trying backup")
        try:
            torch.load(path + ".bak")
            if not model_only:
                return state["model"], state["optimizer"], state["train_list"], state["val_list"], state["epoch"]
            else:
                return state["model"]
        except:
            print(e)
            sys.exit("Backup train state failed, existing")

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