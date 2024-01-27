import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from nvidia.dali.plugin.pytorch import DALIGenericIterator

import dataset
import model
import train
import utils

NUM_EPOCHS = 100
BATCH_SIZE = 32
NUM_WORKERS = 4

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"

OPTIMIZER = "sgd"
LEARNING_RATE = 0.005
MOMENTUM = 0.6

RESUME_ID = None
BACKUP_FREQ = 1
CHECKPOINT_FREQ = 5

ENABLE_DALI = True
ENABLE_BACKUP = True
DATA_AUGMENTATION = True
SPICY_PYTORCH_FLAGS = True

parser = argparse.ArgumentParser(description='Image colorization using UNet')

parser.add_argument('--num_epochs', type=int,
                    default=NUM_EPOCHS,
                    help='Number of training epochs')
parser.add_argument('--batch_size', type=int,
                    default=BATCH_SIZE,
                    help='Batch size for training')
parser.add_argument('--num_workers', type=int,
                    default=NUM_WORKERS,
                    help='Number of workers for data loading')

parser.add_argument('--train_dir', type=str,
                    default=TRAIN_DIR,
                    help='Path to the train image folder')
parser.add_argument('--val_dir', type=str,
                    default=VAL_DIR,
                    help='Path to the validation image folder')

parser.add_argument('--optimizer', type=str,
                    default=OPTIMIZER,
                    help='Optimizer for training',
                    choices=["sgd", "adam"])
parser.add_argument('--learning_rate', type=float,
                    default=LEARNING_RATE,
                    help='Learning rate for the optimizer')
parser.add_argument('--momentum', type=float,
                    default=MOMENTUM,
                    help='Momentum for SGD optimizer')

parser.add_argument('--resume_id', type=str,
                    default=RESUME_ID,
                    help='Wandb run ID to resume training')
parser.add_argument('--backup_freq', type=int,
                    default=BACKUP_FREQ,
                    help='Frequency of model backup during training')
parser.add_argument('--checkpoint_freq', type=int,
                    default=CHECKPOINT_FREQ,
                    help='Frequency of saving checkpoints during training')

parser.add_argument('--enable_dali', type=bool,
                    default=ENABLE_DALI,
                    help='Enable DALI for faster data loading')
parser.add_argument('--enable_backup', type=bool,
                    default=ENABLE_BACKUP,
                    help='Enable model backup during training')
parser.add_argument('--data_augmentation', type=bool,
                    default=DATA_AUGMENTATION,
                    help='Enable data augmentation during training')
parser.add_argument('--spicy_pytorch_flags', type=bool,
                    default=SPICY_PYTORCH_FLAGS,
                    help='Enable spicy PyTorch flags for extra flavor and performance')

args = parser.parse_args()

if args.spicy_pytorch_flags:
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

print("Init dataloader")
if args.enable_dali:
    utils.label_file_check(args.train_dir)
    utils.label_file_check(args.val_dir)

    train_loader = DALIGenericIterator(
    [dataset.ColorDataset.dali_pipeline(image_dir=args.train_dir,
                                        batch_size=args.batch_size,
                                        num_threads=args.num_workers)],
        ['black', 'color'],
        reader_name='Reader'
    )

    val_loader = DALIGenericIterator(
        [dataset.ColorDataset.dali_pipeline(image_dir=args.val_dir,
                                            batch_size=args.batch_size,
                                            num_threads=args.num_workers)],
        ['black', 'color'],
        reader_name='Reader'
    )
else:
    train_dataset = dataset.ColorDataset(args.train_dir, True)
    val_dataset = dataset.ColorDataset(args.val_dir, True)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True)

model = model.UNetResEncoder()

if args.optimizer == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

loss = nn.SmoothL1Loss()

print("Training...")
train.train_model(model, 
                  optimizer, 
                  loss, 
                  train_loader, 
                  val_loader, 
                  "Colorization", 
                  epochs=args.num_epochs, 
                  backup_freq=args.backup_freq if args.enable_backup else -1, 
                  checkpoint_freq=args.checkpoint_freq if args.enable_backup else -1, 
                  resume_id=args.resume_id)
