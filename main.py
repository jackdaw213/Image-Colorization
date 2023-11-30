import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import nvidia_dlprof_pytorch_nvtx

import dataset
import model
import train

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

nvidia_dlprof_pytorch_nvtx.init()

NUM_EPOCHS = 50
BATCH_SIZE = 64

print("Importing dataset")
train_dataset = dataset.ColorDataset('data/train', 'data/train')
val_dataset = dataset.ColorDataset('data/train', 'data/train')

print("Init dataloader")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

model = model.CAE()

#torchinfo.summary(model, (1, 224, 224), batch_dim = 0, col_names = ('input_size', 'output_size', 'num_params', 'kernel_size', 'mult_adds'), verbose = 1)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss = nn.MSELoss()

print("Training...")
with torch.autograd.profiler.emit_nvtx():
  train.train_model(model, train_loader, val_loader, optimizer, loss, NUM_EPOCHS)