import torch
import tqdm as tq
import torchmetrics
from matplotlib import pyplot as plt

import utils

def train_epoch(model, optimizer, loss_func, loader, device):

    model.train()

    epoch_loss = torchmetrics.MeanMetric().to(device)

    for black, color in loader:
        black, color = black.to(device), color.to(device)

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

def train_model(model, optimizer, loss, train_loader, val_loader, epochs, back_up_freq=10, checkpoint_freq=100, load_from_state=False):
                
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    count = 0
    checkpoint_count = 0
    init_epoch = 0
    total_epoch = epochs

    cmodel = torch.compile(model)
    cmodel.to(device)
    train_list = []
    val_list = []
    loss.to(device)

    if load_from_state:
        model_, optimizer_, train_list_, val_list_, epoch_ = utils.load_train_state("model/train.state")
        model.load_state_dict(model_)
        cmodel = torch.compile(model)
        optimizer.load_state_dict(optimizer_)
        train_list = train_list_
        val_list = val_list_
        init_epoch = epoch_
        total_epoch = total_epoch + epoch_ # Add ran epochs to the total amount
        

    for epoch in tq.tqdm(range(epochs), total=total_epoch, desc='Epochs', initial=init_epoch):
        train_loss = train_epoch(cmodel, optimizer, loss, train_loader, device)
        train_list.append(train_loss.cpu())
                
        val_loss = val_epoch(cmodel, loss, val_loader, device)
        val_list.append(val_loss.cpu())

        count = count + 1
        checkpoint_count = checkpoint_count + 1 
        if count == back_up_freq:
            count = 0
            if checkpoint_count == checkpoint_freq:
                print("Save checkpoint at epoch: ", epoch + 1)
                utils.save_train_state(model, optimizer, train_list, val_list, epoch, "model/train.state", True)
                checkpoint_count = 0
            else:
                print("Save train state at epoch: ", epoch + 1)
                utils.save_train_state(model, optimizer, train_list, val_list, epoch, "model/train.state")

    plt.plot(train_list, label='train_loss')
    plt.plot(val_list,label='val_loss')
    plt.legend()
    plt.show()