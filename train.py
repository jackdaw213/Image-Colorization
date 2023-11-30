import torch
import tqdm as tq
import torchmetrics
from matplotlib import pyplot as plt

def train_epoch(model, optimizer, loss_func, loader, device):

    model.train()

    epoch_loss = torchmetrics.MeanMetric().to(device)

    for black, color in loader:
        black, color = black.to(device), color.to(device)

        with torch.cuda.amp.autocast():
            output = model(black)
            loss = loss_func(output, color)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss(loss)

    return epoch_loss.compute()

def val_epoch(model, loss_func, loader, device):

    model.eval()

    epoch_loss = torchmetrics.MeanMetric().to(device)

    for black, color in loader:
        black, color = black.to(device), color.to(device)

        with torch.no_grad():
            output = model(black)
            loss = loss_func(output, color)

        epoch_loss(loss)

    return epoch_loss.compute()

def train_model(model, train_loader, val_loader, optimizer, loss, n_epochs):
                
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    model.to(device)
    loss.cuda()

    train_list = []
    val_list = []

    for epoch in tq.tqdm(range(n_epochs), total=n_epochs, desc='Epochs'):
        train_loss = train_epoch(model, optimizer, loss, train_loader, device)
        train_list.append(train_loss.cpu())
                
        val_loss = val_epoch(model, loss, val_loader, device)
        val_list.append(val_loss.cpu())

    torch.save(model.state_dict(), "model/test")

    plt.plot(train_list, label='train_loss')
    plt.plot(val_list,label='val_loss')
    plt.legend()
    plt.show()