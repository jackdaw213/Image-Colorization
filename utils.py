import torch
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
from skimage.color import rgb2lab
from torchvision.io import read_image
import tqdm as tq
import torch.nn as nn
import torchmetrics
from PIL import Image

def compare_img(img, size=(15, 15)):
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

def l_ab_to_rgb(l, ab):
    img = torch.cat([l.squeeze(0), ab.squeeze(0)], 0)
    img = lab2rgb(img.permute(1, 2, 0))
    return img

def test_learnability(model, image_path, n_epochs):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    img = torch.from_numpy(rgb2lab(read_image(image_path).permute(1, 2, 0))).permute(2, 0, 1).float()
    ab = img[1:, :, :]
    l = img[0, :, :].unsqueeze(0)
    ab = ab.unsqueeze(0).cuda()
    l = l.unsqueeze(0).cuda()

    epoch_loss = torchmetrics.MeanMetric().to(device)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = nn.MSELoss().cuda()

    train_list = []

    model.train()
    for epoch in tq.tqdm(range(n_epochs), total=n_epochs, desc='Epochs'):
        with torch.cuda.amp.autocast():
            output = model(l)
            _loss = loss(output, ab)

        _loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss(_loss)
        train_list.append(epoch_loss.compute().cpu())

    model.eval()
    with torch.no_grad():
        out = model(l)
        result = l_ab_to_rgb(l.cpu(), out.cpu())
        ground = l_ab_to_rgb(l.cpu(), ab.cpu())

        compare_img((Image.open("img/in/0bbRiP.jpg"), result, ground))

    plt.plot(train_list, label='train_loss')
    plt.legend()
    plt.show()