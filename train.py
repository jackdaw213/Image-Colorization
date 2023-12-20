import torch
import tqdm as tq
import torchmetrics
import wandb

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

def train_model(model, optimizer, loss, train_loader, val_loader, project_name, epochs, back_up_freq=10, checkpoint_freq=100, resume_id=None):
                
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    count = 0
    checkpoint_count = 0
    init_epoch = 0
    loss.to(device)

    config = {
    "model": model.__class__.__name__,
    "optimizer": optimizer.__class__.__name__,
    "loss": loss.__class__.__name__,
    "learning_rate": optimizer.param_groups[0]['lr'],
    "momentum": optimizer.param_groups[0]['momentum'],
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
        train_loss = train_epoch(cmodel, optimizer, loss, train_loader, device)
        val_loss = val_epoch(cmodel, loss, val_loader, device)
        wandb.log({"loss": train_loss, "loss_val": val_loss, "epoch": epoch})
        
        count = count + 1
        checkpoint_count = checkpoint_count + 1 
        if count == back_up_freq:
            count = 0
            if checkpoint_count == checkpoint_freq:
                utils.save_train_state(model, optimizer, epoch, "model/train.state", True)
                checkpoint_count = 0
                print("Saved checkpoint at epoch: ", epoch)
            else:
                utils.save_train_state(model, optimizer, epoch, "model/train.state")
                print("Saved train state at epoch: ", epoch)
    run.finish()