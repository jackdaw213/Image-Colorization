import torch
import torch.optim as optim
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import wandb
import tqdm as tq

import dataset
import model as model_module
import train as train_module
import utils
import model_parts

BATCH_SIZE = 32
EPOCHS = 75
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
PROJECT_NAME = "unet-resnet-sweep"

def build_optimizer(model, config):
    if config.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    elif config.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    return optimizer

def build_model():
    model = model_module.UNetResEncoder()
    model = torch.compile(model)
    return model.cuda()

def build_loader(batch_size):
    utils.label_file_check(TRAIN_DIR)
    utils.label_file_check(VAL_DIR)

    train_loader = DALIGenericIterator(
    [dataset.ColorDataset.dali_pipeline(image_dir=TRAIN_DIR,
                                        batch_size=batch_size,
                                        num_threads=4)],
        ['black', 'color'],
        reader_name='Reader'
    )

    val_loader = DALIGenericIterator(
        [dataset.ColorDataset.dali_pipeline(image_dir=VAL_DIR,
                                            batch_size=batch_size,
                                            num_threads=4)],
        ['black', 'color'],
        reader_name='Reader'
    )
    return train_loader, val_loader

def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        loss = model_parts.HuberLoss().cuda()
        model = build_model()
        train, val = build_loader(BATCH_SIZE)
        optimizer = build_optimizer(model, config)

        for epoch in tq.tqdm(range(EPOCHS), total=EPOCHS, desc='Epochs'):
            avg_loss = train_module.train_epoch(model, optimizer, loss, train, "cuda:0")  
            val_loss = train_module.val_epoch(model, loss, val, "cuda:0")
            wandb.log({"loss": avg_loss, "loss_val": val_loss, "epoch": epoch})

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

sweep_id = wandb.sweep(sweep=sgd, project=PROJECT_NAME)
wandb.agent(sweep_id, train, project=PROJECT_NAME)

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

sweep_id = wandb.sweep(sweep=adam, project=PROJECT_NAME)
wandb.agent(sweep_id, train)

