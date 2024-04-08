import torch
import tqdm as tq
import torchmetrics
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from datetime import datetime
import wandb

import utils
from model import UNetResEncoder

def train_color(model, optimizer, loss_func, loader, device):

    model.train()

    epoch_loss = torchmetrics.MeanMetric().to(device)

    for _, data in enumerate(loader):
        if isinstance(loader, DALIGenericIterator):
            black, color, mask = data[0]["black"], data[0]["color"], data[0]["mask"].unsqueeze(dim=1)
        else:
            black, color, mask = data
            black, color = black.to(device), color.to(device), mask.unsqueeze(dim=1).to(device)

        output = model(black)
        loss = loss_func(output, color, mask)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss(loss)

    return epoch_loss.compute()

def val_color(model, loss_func, loader, device):

    model.eval()

    epoch_loss = torchmetrics.MeanMetric().to(device)

    for _, data in enumerate(loader):
        if isinstance(loader, DALIGenericIterator):
            black, color, mask = data[0]["black"], data[0]["color"], data[0]["mask"].unsqueeze(dim=1).to(device)
        else:
            black, color, mask = data
            black, color = black.to(device), color.to(device), mask.unsqueeze(dim=1).to(device)

        with torch.no_grad():
            output = model(black)
            loss = loss_func(output, color, mask)

        epoch_loss(loss)

    return epoch_loss.compute()

def train_style(model, optimizer, loss_func, loader, device):

    model.train()

    epoch_loss = torchmetrics.MeanMetric().to(device)

    for _, data in enumerate(loader):
        pass

    return epoch_loss.compute()

def val_style(model, loss_func, loader, device):

    model.eval()

    epoch_loss = torchmetrics.MeanMetric().to(device)

    for _, data in enumerate(loader):
        pass

    return epoch_loss.compute()

def train_model(model, optimizer, loss, train_loader, val_loader, project_name, epochs, checkpoint_freq, resume_id):
                
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    checkpoint_count = 0
    init_epoch = 0
    loss.to(device)

    config = {
    "model": model.__class__.__name__,
    "optimizer": optimizer.__class__.__name__,
    "loss": loss.__class__.__name__,
    "learning_rate": optimizer.param_groups[0]['lr'],
    "momentum": None if isinstance(optimizer, torch.optim.Adam) else optimizer.param_groups[0]['momentum'],
    }

    if resume_id is not None:
        model_, optimizer_, epoch_ = utils.load_train_state("model/train.state")

        model.load_state_dict(model_)
        cmodel = torch.compile(model, mode="reduce-overhead")
        cmodel.to(device)

        optimizer.load_state_dict(optimizer_)

        init_epoch = epoch_ + 1 # PLus 1 means start at the next epoch
        run = wandb.init(project=project_name, config=config, id=resume_id, resume=True)
    else:
        cmodel = torch.compile(model, mode="reduce-overhead")
        cmodel.to(device)
        run = wandb.init(project=project_name, config=config)

    for epoch in tq.tqdm(range(init_epoch, epochs), total=epochs, desc='Epochs', initial=init_epoch):
        if isinstance(model, UNetResEncoder):
            train_loss = train_color(cmodel, optimizer, loss, train_loader, device)
            val_loss = val_color(cmodel, loss, val_loader, device)
        else:
            train_loss = train_style(cmodel, optimizer, loss, train_loader, device)
            val_loss = val_style(cmodel, loss, val_loader, device)
            
        wandb.log({"loss": train_loss, "loss_val": val_loss, "epoch": epoch})
        
        checkpoint_count = checkpoint_count + 1 
        if checkpoint_count == checkpoint_freq:
            utils.save_train_state(model, optimizer, epoch, "model/train.state")
            checkpoint_count = 0
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Saved checkpoint at epoch: {epoch + 1} ({now})")

    run.finish()