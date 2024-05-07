import torch
import tqdm as tq
import torchmetrics
from datetime import datetime
import wandb

import utils
from model import UNetResEncoder

"""
Yet another mini-blog, today's dish is mixed precision training! I actually tried this wayyy back at the 
beginning of the project however 1 problem, NaN loss. Even with GradScaler, after just a few epochs,
NaN values pop up everywhere making it impossible to train the model so I scraped it. But now I did a
bit more digging and found these. First, AMP is what the cool kids are doing now [1*] even google with
their Coral TPU [2*]. Second, a redditor recommends using BF16 if your hardware supports it [3*] (31 upvotes
so seems kinda trustworthy) because it offers better stability than FP16. Also I even found more people
who have the same NaN loss problem as me and there are some ways to prevent it [4*][5*]. However I didn't
implement any of these because my GPU supports BF16

And the results are amazing! Training time is reduced from 1h30 to 50 minutes per epoch and memory usage drops
about 1-2GB and importantly NO NaN loss. Also it's quite important to optimize the training process with the
right tools. I remember in the early days of the project, unoptimized model, standard DataLoader instead of 
DALI, no AMP, no pretrained encoder, and the training time was about 3 DAYS. Now it's down to 8.5 hours 

[1*]: https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/#best-practices
[2*]: https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus
[3*]: https://www.reddit.com/r/MachineLearning/comments/vndtn8/d_mixed_precision_training_difference_between/
[4*]: https://discuss.pytorch.org/t/adam-half-precision-nans/1765/5
[5*]: https://discuss.pytorch.org/t/nan-loss-with-torch-cuda-amp-and-crossentropyloss/108554/19
"""

def train_color(model, optimizer, scaler, loss_func, loader, device, args):

    model.train()

    epoch_loss = torchmetrics.MeanMetric().to(device)

    if args.amp_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    for _, data in enumerate(loader):
        if args.enable_dali:
            black, color, mask = data[0]["black"], data[0]["color"], data[0]["mask"].unsqueeze(dim=1).to(device)
        else:
            black, color, mask = data
            black, color = black.to(device), color.to(device), mask.unsqueeze(dim=1).to(device)

        with torch.autocast(device_type="cuda", dtype=dtype, enabled=args.enable_amp):
            output = model(black)
            loss = loss_func(output, color, mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        epoch_loss(loss)

    return epoch_loss.compute()

def val_color(model, loss_func, loader, device, args):

    model.eval()

    epoch_loss = torchmetrics.MeanMetric().to(device)

    if args.amp_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    for _, data in enumerate(loader):
        if args.enable_dali:
            black, color, mask = data[0]["black"], data[0]["color"], data[0]["mask"].unsqueeze(dim=1).to(device)
        else:
            black, color, mask = data
            black, color = black.to(device), color.to(device), mask.unsqueeze(dim=1).to(device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=args.enable_amp):
            output = model(black)
            loss = loss_func(output, color, mask)

        epoch_loss(loss)

    return epoch_loss.compute()

def train_style(model, optimizer, scaler, loss_func, loader, device, args):

    model.train()

    content_metric = torchmetrics.MeanMetric().to(device)
    style_metric = torchmetrics.MeanMetric().to(device)
    total_metric = torchmetrics.MeanMetric().to(device)

    if args.amp_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    for _, data in enumerate(loader):
        content = data[0]['content']

        with torch.autocast(device_type="cuda", dtype=dtype, enabled=args.enable_amp):
            content_out, content_features, content_features_loss = model(content)
            content_loss, style_loss = loss_func(content, content_out, content_features, content_features_loss)
            total_loss = content_loss + style_loss

        # https://discuss.pytorch.org/t/whats-the-correct-way-of-using-amp-with-multiple-losses/93328/3
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        content_metric(content_loss)
        style_metric(style_loss)
        total_metric(total_loss)

    return content_metric.compute(), style_metric.compute(), total_metric.compute()

def val_style(model, loss_func, loader, device, args):

    model.eval()

    content_metric = torchmetrics.MeanMetric().to(device)
    style_metric = torchmetrics.MeanMetric().to(device)
    total_metric = torchmetrics.MeanMetric().to(device)
    
    if args.amp_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    for _, data in enumerate(loader):
        content = data[0]['content']

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=args.enable_amp):
            content_out, content_features, content_features_loss = model(content)
            content_loss, style_loss = loss_func(content, content_out, content_features, content_features_loss)
            total_loss = content_loss + style_loss

        content_metric(content_loss)
        style_metric(style_loss)
        total_metric(total_loss)

    return content_metric.compute(), style_metric.compute(), total_metric.compute()

def train_model(model, optimizer, loss, train_loader, val_loader, args):
                
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    checkpoint_count = 0
    init_epoch = 0
    loss.to(device)
    project_name = "Colorization" if args.model == "color" else "StyleTransfer"
    scaler = torch.cuda.amp.GradScaler(enabled=args.enable_amp)

    config = {
    "model": model.__class__.__name__,
    "optimizer": optimizer.__class__.__name__,
    "loss": loss.__class__.__name__,
    "learning_rate": optimizer.param_groups[0]['lr'],
    "momentum": None if isinstance(optimizer, torch.optim.Adam) else optimizer.param_groups[0]['momentum'],
    }

    if args.resume_id is not None:
        model_, optimizer_, scaler_, epoch_ = utils.load_train_state("model/train.state")

        model.load_state_dict(model_)
        cmodel = torch.compile(model, mode="reduce-overhead")
        cmodel.to(device)

        optimizer.load_state_dict(optimizer_)
        scaler.load_state_dict(scaler_)

        init_epoch = epoch_ + 1 # PLus 1 means start at the next epoch
        run = wandb.init(project=project_name, config=config, id=args.resume_id, resume=True)
    else:
        cmodel = torch.compile(model, mode="reduce-overhead")
        cmodel.to(device)
        run = wandb.init(project=project_name, config=config)

    for epoch in tq.tqdm(range(init_epoch, args.epochs), total=args.epochs, desc='Epochs', initial=init_epoch):
        if args.model == "color":
            train_loss = train_color(cmodel, optimizer, scaler, loss, train_loader, device, args)
            val_loss = val_color(cmodel, loss, val_loader, device, args)
            wandb.log({"loss": train_loss, "loss_val": val_loss, "epoch": epoch})
        else:
            train_loss = train_style(cmodel, optimizer, scaler, loss, train_loader, device, args)
            val_loss = val_style(cmodel, loss, val_loader, device, args)
            wandb.log({"content_loss": train_loss[0], 
                       "style_loss": train_loss[1], 
                       "total_loss": train_loss[2], 
                       "content_loss_val": val_loss[0], 
                       "style_loss_val": val_loss[1], 
                       "total_loss_val": val_loss[2], 
                       "epoch": epoch})
        
        checkpoint_count = checkpoint_count + 1 
        if checkpoint_count == args.checkpoint_freq:
            utils.save_train_state(model, optimizer, scaler, epoch, "model/train.state")
            checkpoint_count = 0
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Saved checkpoint at epoch: {epoch + 1} ({now})")

    if args.model == "color":
        model_scripted = torch.jit.trace(model.cpu(), torch.rand(1,3,256,256))
        model_scripted.save('model/model_color.pt')
    else:
        model_scripted = torch.jit.trace(model.cpu(), (torch.rand(1,3,256,256),torch.rand(1,3,256,256)))
        model_scripted.save('model/model_style.pt')
    run.finish()
