import torch
from torch.utils.data import DataLoader
import nvidia_dlprof_pytorch_nvtx

import dataset
import model
import train
import auto_parts

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

#nvidia_dlprof_pytorch_nvtx.init()

NUM_EPOCHS = 100
BATCH_SIZE = 8

print("Importing dataset")
train_dataset = dataset.ColorDataset('data/train', True)
val_dataset = dataset.ColorDataset('data/val', True)

print("Init dataloader")
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=4, 
    pin_memory=True)
val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=4, 
    pin_memory=True)

model = model.UNet()
model._initialize_weights()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.6)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

#loss = nn.MSELoss().cuda()
loss = auto_parts.HuberLoss().cuda()

#torchinfo.summary(model, (1, 224, 224), batch_dim = 0, col_names = ('input_size', 'output_size', 'num_params', 'kernel_size', 'mult_adds'), verbose = 1)

print("Training...")
#with torch.autograd.profiler.emit_nvtx():
train.train_model(model, 
                  optimizer, 
                  loss, 
                  train_loader, 
                  val_loader, 
                  "Colorization", 
                  epochs=NUM_EPOCHS, 
                  back_up_freq=1, 
                  checkpoint_freq=5)
