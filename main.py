import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from nvidia.dali.plugin.pytorch import DALIGenericIterator

import dataset
import model
import train
import model_parts as mp

NUM_EPOCHS = 10
BATCH_SIZE = 8
NUM_WORKERS = 4

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"

OPTIMIZER = "adam"
LEARNING_RATE = 0.00005
MOMENTUM = 0.6

RESUME_ID = None
CHECKPOINT_FREQ = 1

AMP_TYPE = "bf16"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")

parser = argparse.ArgumentParser(description='Image colorization using UNet')

parser.add_argument('-cp', '--color_peak', action='store_true',
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

parser.add_argument('--enable_dali', action='store_true',
                    help='Enable DALI for faster data loading')
parser.add_argument('--enable_amp', action='store_true',
                    help='Enable Mixed Precision for faster training and lower memory usage')

parser.add_argument('-ampt', '--amp_dtype', type=str,
                    default=AMP_TYPE,
                    help='Set dtype for amp',
                    choices=["bf16", "fp16"])

args = parser.parse_args()

print("Init dataloader")
if args.enable_dali:
    train_loader = DALIGenericIterator(
        [dataset.ColorDataset.dali_pipeline(image_dir=args.train_dir,
                                            color_peak=args.color_peak,
                                            batch_size=args.batch_size,
                                            num_threads=args.num_workers,
                                            prefetch_queue_depth=4 if args.enable_amp else 2)],
        ['black', 'color', 'mask'],
        reader_name='Reader'
    )

    val_loader = DALIGenericIterator(
        [dataset.ColorDataset.dali_pipeline(image_dir=args.val_dir,
                                            color_peak=args.color_peak,
                                            batch_size=args.batch_size,
                                            num_threads=args.num_workers,
                                            prefetch_queue_depth=4 if args.enable_amp else 2)],
        ['black', 'color', 'mask'],
        reader_name='Reader'
    )
else:
    train_dataset = dataset.ColorDataset(args.train_dir, True, args.color_peak)
    val_dataset = dataset.ColorDataset(args.val_dir, True, args.color_peak)

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
loss = mp.ColorLoss()

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
                args)

"""
Today is 04/04/2024 and I'm thinking about the paper I'm trying to implement. 
I see that it was published in 2017 and think to myself: "Ha, this paper was 
published quite recently" only for a truck to hit my brain and realized that 
2017 was 7 FREAKING years ago, WTF i thought it was 3-4 years ago. Where did 
all of those years go?. I was in 8th grade and now I'm in second year of 
college ?? I swear I probably got into an accident and in coma for 3 years 
or something cause I was 18 YESTERDAY and now I'm 20!! 
I'm probably retiring next year if this keep happening :((
⣿⣿⣿⠛⢻⡟⠛⠛⠛⠛⠛⣿⠋⢻⣿⠟⠛⠋⠛⢿⣿⣿⣿⣿⣟⠛⢻⣿⡿⠛⠛⠛⠛⢿⣿⡟⠛⠛⠛⠛⡟⠛⢿⣿⣿⠛⢛⡿⠛⠛⠛⠛⣿⠛⠛⠛⠛⠻⣿⣿
⣿⣿⣿⠀⢸⣷⣷⡇⠀⣾⣾⣿⣤⣽⣇⠀⠰⢿⣷⣿⣿⣿⣿⣿⣗⠀⢸⡟⠀⢰⣾⣶⡆⠀⢻⡇⠀⢼⢾⢾⣷⠀⠘⣿⡏⠀⣼⣯⠀⢰⢷⢷⣿⠂⠀⣿⡆⠀⣸⣿
⣿⣿⣿⠀⢸⣿⣿⡇⠀⣿⣿⣿⣿⣿⣿⣦⣤⡀⠈⢻⣿⣿⣿⣿⣗⠀⢸⡇⠀⢻⣿⣿⡯⠀⢸⡇⠀⢤⣤⣼⣿⣧⠀⢻⠁⢰⣿⡷⠀⢠⣤⣬⣿⠂⠀⣀⠀⠲⣿⣿
⣿⣿⣿⠀⢸⣿⣿⡇⠀⣿⣿⣿⣿⣿⡇⠉⠛⠉⢀⣼⣿⣿⡯⠙⠁⢀⣾⣿⣄⡀⠉⠋⢀⣠⣿⡇⠀⠉⠋⠋⣿⣿⡄⠀⢀⣿⣿⣟⠀⠈⠋⠋⣿⠂⠀⣿⣧⠀⠹⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢿⢻⢕⣗⣗⢖⢍⠝⠽⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢿⣿⣿⢿⣿⡿⣿⣻⣿⣻⡽⡛⡜⡜⡮⡳⣳⢕⡯⡮⡪⡂⡊⢳⣿⣿⣽⣿⣟⣿⣿⣻⣿⣻⣿⣟⣿⣿⣻⣿⡿⣿⣿⢿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⢿⣿⣿⣽⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢏⢎⢎⢮⡳⡽⡽⣺⡳⡧⡳⣕⢕⡪⡐⢽⡿⣟⣿⣿⢿⣿⣿⣿⣿⣿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷
⣿⣿⣿⡿⣿⣽⣷⣿⣿⣿⣿⣿⣿⣿⢿⣻⣽⣿⣯⣿⣿⣽⣾⣜⢵⢝⡷⡽⣺⣝⢮⢏⡯⡯⣮⣳⢨⢈⢦⣿⣿⣿⣿⣿⣿⣷⣿⣷⣿⣿⣿⣷⣿⣿⣾⣿⣿⣻⣿⣿
⣾⣿⣷⣿⣿⣿⣿⣿⣿⣿⣿⣾⣿⣾⣿⣿⣿⡿⣟⣿⣽⣿⡻⣜⢕⡯⡮⣟⡯⣯⡳⣝⣞⢽⢕⢧⣓⢅⣿⣿⣯⣷⣿⣷⣿⣯⣿⣟⣿⣿⣽⣿⣯⣿⣿⣽⣿⣿⢿⣿
⣿⣿⣿⣿⣿⣿⢿⣟⣿⣽⣾⣿⣟⣿⣟⣯⣿⣿⣿⢿⣋⢉⠪⡮⣳⢯⣟⢼⢯⢗⣟⡮⡪⣳⣝⢕⢎⣾⣿⣷⣿⢿⣻⣽⣿⣽⣿⢿⣻⣯⣿⣿⣽⣿⣯⣿⣿⣻⣿⣿
⣷⣿⣿⣾⣿⣾⣿⣿⣿⣿⡿⣟⣿⡿⣿⣿⣻⡷⣿⣻⣯⣧⡕⢌⢳⣟⡾⣽⣫⢯⡺⣝⢎⢊⢣⢫⣾⣿⣷⢿⣻⣿⡿⣿⣻⣯⣿⣿⣿⣿⣟⣯⣿⣯⣿⣟⣿⣿⣻⣷
⣿⣿⡿⣟⣿⣟⣿⣟⣯⣷⣿⣿⡿⣿⣿⣻⣽⣻⣽⣻⡾⣯⣿⣆⢅⢻⡽⡷⣽⢮⡻⡜⢮⣎⢇⣿⡿⣷⣿⣿⢿⣻⣿⡿⣿⣿⣻⣽⣾⣿⣻⣿⣟⣿⣟⣿⣿⣟⣿⣿
⣿⣷⣿⣿⣿⡿⣿⡿⣿⣟⣯⣷⣿⣟⡾⣷⢿⣟⣿⣽⣿⡽⣾⣻⣦⠑⣟⡯⡯⣗⠽⣽⣵⣿⣿⣽⡿⣿⡷⣿⣿⢿⣯⣿⣿⣯⣿⣿⣻⣽⣿⣻⣿⣻⣿⣟⣯⣿⣿⣷
⣿⣿⣻⣿⣷⣿⣿⢿⣿⢿⣟⣿⣳⡿⣽⣿⣿⣿⢿⣻⣾⣟⣯⢷⣟⣧⢸⣫⣯⣷⡿⣿⢷⡿⣷⢿⣟⣿⣻⣿⣽⣿⢿⣿⣿⣾⣿⣽⡿⣟⣿⣟⣿⣿⣻⣿⡿⣟⣿⣾
⣻⣽⣿⣿⣽⣿⣾⣿⡿⣿⡿⣽⣯⣿⣟⣿⣷⣿⣿⡿⣿⣯⡿⣯⢷⣻⣯⣷⣻⣽⡿⣟⣿⢿⣻⣿⣻⣟⣯⣿⡾⣟⣿⣿⣾⢯⣷⣿⡿⣿⣿⣻⣿⣻⣿⣻⣿⣿⡿⣿
⣿⣿⣻⣽⣿⣷⣿⣷⣿⢿⣻⣿⢷⣿⣿⢿⣻⣿⣽⣿⢿⣿⡽⣟⣯⢿⣺⡽⣎⣿⣟⣿⣻⡿⣯⣿⣯⣿⣻⣷⡿⣟⣿⡾⣟⣿⡿⣽⣿⣿⣽⣿⣻⣿⣻⣿⣯⣷⣿⣿
⣽⣿⡿⣟⣿⣾⣿⣾⢿⣟⣿⣿⣻⣿⣻⣿⣿⣻⣯⣿⢿⣿⣟⣯⣿⡯⣷⣻⡽⡷⣟⣯⣿⣻⣯⣷⡿⣾⣟⣷⡿⣟⣿⣻⣟⣯⣿⣯⣷⡿⣯⣿⣟⣿⣻⣽⣿⣽⣿⣽
⣿⣻⣿⡿⣿⡿⣾⡿⣿⢿⣟⣿⡿⣿⡿⣿⣾⡿⣟⣿⣻⣿⣟⣾⣿⣯⡯⣷⣻⣽⢿⣯⣿⣽⣷⢿⣻⣯⣿⣽⡿⣟⣿⣻⣟⣯⣷⣿⣷⣽⢿⣻⣿⣻⣿⣟⣯⣿⣟⣿
⣻⣿⣻⣿⡿⣿⣿⣻⣿⣿⣿⣿⢿⣿⣿⢿⣷⣿⡿⣿⣻⣾⣿⣽⣿⣷⢿⣽⣳⡯⣿⡷⣟⣷⣿⣻⣿⣽⣷⢿⣻⣿⣻⣟⣿⣯⣿⣾⣷⣿⢿⣻⣿⣻⣽⣿⣻⣿⣻⣿
⢿⣻⣿⣟⣿⣿⣽⡿⣷⡿⣿⣾⣿⣿⣾⣿⣿⣾⣿⣻⣯⣷⣟⣿⣿⣟⣿⣞⣗⣿⢽⣿⣻⣽⡾⣿⢾⣷⢿⣟⣿⣽⣿⣽⡿⣾⢿⣾⢷⣿⢿⣿⣽⢿⣻⣿⣻⣿⣻⣿
⣿⣿⣟⣿⣿⣽⣿⣻⣿⢿⣿⣿⣯⣿⣿⡿⣿⣻⣿⣿⣞⣿⢿⣿⣿⣻⣷⢿⣞⣞⣯⣿⢯⣷⡿⣟⣿⣽⡿⣯⡿⣷⡿⣾⡿⣿⣟⣿⣟⣿⣟⣿⣾⣿⢿⣻⣿⣿⣿⡿
⣿⣯⣿⣿⣽⣿⣽⣿⣻⣿⣿⡿⣿⣾⣿⣿⢿⣿⢿⣻⡿⣯⣿⣿⣿⣻⣿⣻⣽⡾⣽⣾⢿⡷⣿⣻⣯⣷⡿⣟⣿⣯⣿⣟⣿⣯⣿⣯⣿⣯⣿⣯⣷⣿⣿⣿⣷⣿⣿⣿
⣿⣻⣯⣿⣟⣯⣿⣟⣿⣽⣷⣿⣿⣿⣿⣾⣿⣿⣿⡿⣟⣿⣾⣿⣿⣟⣯⡿⣞⣿⣯⣿⣟⣿⣟⣯⣿⡾⣿⣻⡿⣾⡷⣿⣻⣾⣷⡿⣷⣿⣷⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⡿⣿⡿⣿⡿⣿⣻⣿⣻⣿⣿⣿⣿⣽⣿⣿⣿⣿⣿⣿⡿⣿⣽⣾⣿⢿⣟⣿⣺⣷⣿⣯⣷⡿⣯⣷⡿⣟⣿⣻⣟⣿⣿⣟⣿⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⢿⣿⣿⣿⣿⡿⣿⣟⣿⣿⣷⣿⣿⣿⣿⣽⣾⣿⣿⣿⣿⣿⣿⣯⣿⣿⣿⣿⣿⣿⣿⣽⣾⢿⣻⣷⡿⣿⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⢿⣷⣿⣷⣿⣿⣿⢿⣷⣿⣿⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣽⣿⣿⣿⣿⣿⣿⣿⣿⡿⣾⡿⣟⣷⣿⢿⣿⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣾⣿⣿⣿⣽⣿⣷⣿⣿⣿⢿⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣽⣟⡿⣾⣻⣻⣽⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣟⣿⣯⣿⣾⣿⣿⣾⡿⣟⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢇⣯⣻⣺⣞⣷⣿⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⡿⣿⡿⣿⣾⣿⣿⣿⣿⣿⣿⣿⣿⣷⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣽⣽⣾⣾⡿⣿⣿⣿⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
"""
