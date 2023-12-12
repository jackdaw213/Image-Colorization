import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import tqdm as tq

import dataset
import model as model_module
import train as train_module
import auto_parts

BATCH_SIZE = 8
EPOCHS = 75

def build_optimizer(model, config):
    if config.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    elif config.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    return optimizer

def build_model():
    model = model_module.UNet()
    model._initialize_weights()
    model = torch.compile(model)
    return model.cuda()

def build_loader(batch_size):
    train_dataset = dataset.ColorDataset('data/train_small3', True)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True)
    return train_loader

def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        loss = auto_parts.HuberLoss().cuda()
        model = build_model()
        loader = build_loader(BATCH_SIZE)
        optimizer = build_optimizer(model, config)

        for epoch in tq.tqdm(range(EPOCHS), total=EPOCHS, desc='Epochs'):
            avg_loss = train_module.train_epoch(model, optimizer, loss, loader, "cuda:0")
            wandb.log({"loss": avg_loss, "epoch": epoch})           

wandb.login()

# Can't figured out how to ignore momentum when using adam so I just split the sweep
# into 2 parts, one with adam + set momentum to 0 and the other with sgd and default
# momentum list
sgd = {
    'method': 'grid',
    'metric': {
        'name': 'loss',
        'goal': 'minimize'   
    },
    'name': 'unet-sweep-sgd',
    'parameters': {
        'learning_rate': {'values': [0.00001, 0.0005, 0.0001, 0.001, 0.01, 0.1]},
        'momentum': {'values': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]},
        'optimizer': {'values': ["sgd"]}
    }
}

sweep_id = wandb.sweep(sweep=sgd, project='unet-sweep')
wandb.agent(sweep_id, train)

adam = {
    'method': 'grid',
    'metric': {
        'name': 'loss',
        'goal': 'minimize'   
    },
    'name': 'unet-sweep-adam',
    'parameters': {
        'learning_rate': {'values': [0.00001, 0.0005, 0.0001, 0.001, 0.01, 0.1]},
        'optimizer': {'values': ["adam"]}
    }
}

sweep_id = wandb.sweep(sweep=adam, project='unet-sweep')
wandb.agent(sweep_id, train)

