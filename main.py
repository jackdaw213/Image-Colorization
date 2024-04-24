import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from nvidia.dali.plugin.pytorch import DALIGenericIterator

import dataset
import model
import train
import auto_parts as ap
import utils

MODEL = "color"
COLOR_PEAK = True
NUM_EPOCHS = 10
BATCH_SIZE = 8
NUM_WORKERS = 4

TRAIN_DIR = "data/train_color"
VAL_DIR = "data/val_color"
TRAIN_DIR_CONTENT = "data/train_content"
VAL_DIR_CONTENT = "data/val_content"
TRAIN_DIR_STYLE = "data/train_style"
VAL_DIR_STYLE = "data/val_style"

OPTIMIZER = "adam"
LEARNING_RATE = 0.00005
MOMENTUM = 0.6

RESUME_ID = None
CHECKPOINT_FREQ = 5

ENABLE_DALI = True
DATA_AUGMENTATION = True
SPICY_PYTORCH_FLAGS = True

parser = argparse.ArgumentParser(description='Image colorization using UNet')

parser.add_argument('-m', '--model', type=str,
                    default=MODEL,
                    help='Select what model to train',
                    choices=["color", "style"])
parser.add_argument('-cp', '--color_peak', type=bool,
                    default=COLOR_PEAK,
                    help='Passing color infos to the model during training')
parser.add_argument('-e', '--epochs', type=int,
                    default=NUM_EPOCHS,
                    help='Number of training epochs')
parser.add_argument('-bs', '--batch_size', type=int,
                    default=BATCH_SIZE,
                    help='Batch size for training')
parser.add_argument('-nw' ,'--num_workers', type=int,
                    default=NUM_WORKERS,
                    help='Number of workers for data loading')

parser.add_argument('-td', '--train_dir', type=str,
                    default=TRAIN_DIR,
                    help='Path to the color model train image folder')
parser.add_argument('-vd', '--val_dir', type=str,
                    default=VAL_DIR,
                    help='Path to the color model validation image folder')

parser.add_argument('-tdc', '--train_dir_content', type=str,
                    default=TRAIN_DIR_CONTENT,
                    help='Path to the style model train_content image folder')
parser.add_argument('-vdc', '--val_dir_content', type=str,
                    default=VAL_DIR_CONTENT,
                    help='Path to the style model val_content image folder')

parser.add_argument('-tds', '--train_dir_style', type=str,
                    default=TRAIN_DIR_STYLE,
                    help='Path to the style model train_style image folder')
parser.add_argument('-vds', '--val_dir_style', type=str,
                    default=VAL_DIR_STYLE,
                    help='Path to the style model val_style image folder')

parser.add_argument('-op', '--optimizer', type=str,
                    default=OPTIMIZER,
                    help='Optimizer for training',
                    choices=["sgd", "adam"])
parser.add_argument('-lr', '--learning_rate', type=float,
                    default=LEARNING_RATE,
                    help='Learning rate for the optimizer')
parser.add_argument('--momentum', type=float,
                    default=MOMENTUM,
                    help='Momentum for SGD optimizer')

parser.add_argument('-id', '--resume_id', type=str,
                    default=RESUME_ID,
                    help='Wandb run ID to resume training')
parser.add_argument('-cf', '--checkpoint_freq', type=int,
                    default=CHECKPOINT_FREQ,
                    help='Frequency of saving checkpoints during training, -1 for no checkpoints')

parser.add_argument('-dali', '--enable_dali', type=bool,
                    default=ENABLE_DALI,
                    help='Enable DALI for faster data loading')
parser.add_argument('-da', '--data_augmentation', type=bool,
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
    if args.model == "color":
        train_loader = DALIGenericIterator(
            [dataset.ColorDataset.dali_pipeline(image_dir=args.train_dir,
                                                color_peak=args.color_peak,
                                                batch_size=args.batch_size,
                                                num_threads=args.num_workers)],
            ['black', 'color', 'mask'],
            reader_name='Reader'
        )

        val_loader = DALIGenericIterator(
            [dataset.ColorDataset.dali_pipeline(image_dir=args.val_dir,
                                                color_peak=args.color_peak,
                                                batch_size=args.batch_size,
                                                num_threads=args.num_workers)],
            ['black', 'color', 'mask'],
            reader_name='Reader'
        )
    else:
        train_loader = DALIGenericIterator(
            [dataset.StyleDataset.dali_pipeline(content_dir=args.train_dir_content,
                                                style_dir=args.train_dir_style,
                                                batch_size=args.batch_size,
                                                num_threads=args.num_workers)],
            ['content', 'style'],
            reader_name='Reader'
        )

        val_loader = DALIGenericIterator(
            [dataset.StyleDataset.dali_pipeline(content_dir=args.val_dir_content,
                                                style_dir=args.val_dir_style,
                                                batch_size=args.batch_size,
                                                num_threads=args.num_workers)],
            ['content', 'style'],
            reader_name='Reader'
        )
else:
    if args.model == "color":
        train_dataset = dataset.ColorDataset(args.train_dir, True, args.color_peak)
        val_dataset = dataset.ColorDataset(args.val_dir, True, args.color_peak)
    else:
        pass

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
    
if args.model == "color":
    model = model.UNetResEncoder()
    loss = ap.ColorLoss()
else:
    model = model.StyleTransfer()
    loss = ap.AdaINLoss()

if args.optimizer == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

print("Training...")
train.train_model(model, 
                  optimizer, 
                  loss, 
                  train_loader, 
                  val_loader, 
                  "Colorization" if args.model == "color" else "StyleTransfer", 
                  epochs=args.epochs, 
                  checkpoint_freq=args.checkpoint_freq, 
                  resume_id=args.resume_id)
