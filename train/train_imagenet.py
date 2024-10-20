import os
import random
import warnings
import numpy as np
from tqdm import tqdm
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data.dataloader import default_collate
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets.samplers import DistributedSampler, RandomClipSampler, UniformClipSampler
from torch.utils.tensorboard import SummaryWriter
from spikingjelly.activation_based import functional, neuron
import utils
import transforms
from model.SEWResNet import *
from model.ShallowSEWResNet import *
from model.SpikingMobileNet import *


_seed_ = 2022


def get_args():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="/dataset/ImageNet2012/", help="dataset path")
    parser.add_argument("--cache-dataset", action="store_true", help="Cache the datasets for quicker initialization. It also serializes the transforms")
    
    parser.add_argument("--epochs", default=320, type=int, help="number of epochs to train")
    parser.add_argument("--batch-size", default=32, type=int, help="number of images per gpu")
    
    parser.add_argument("--opt", default="sgd", type=str, choices=["sgd", "adam"], help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight-decay", default=0, type=float, help="weight decay (L2 penalty)")
    
    parser.add_argument("--train-crop-size", default=224, type=int, help="the random crop size used for training")
    parser.add_argument("--val-resize-size", default=256, type=int, help="the resize size used for validation")
    parser.add_argument("--val-crop-size", default=224, type=int, help="the center crop size used for validation")
    
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy")
    parser.add_argument("--random-erase", default=0, type=float, help="random erasing probability")
    parser.add_argument("--mixup-alpha", default=0, type=float, help="mixup alpha")
    parser.add_argument("--cutmix-alpha", default=0, type=float, help="cutmix alpha")
    parser.add_argument("--label-smoothing", default=0, type=float, help="label smoothing")
    
    parser.add_argument("--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters")
    parser.add_argument("--model-ema-steps", default=32, type=int, help="the number of iterations that controls how often to update the EMA model")
    parser.add_argument("--model-ema-decay", default=0.99998, type=float, help="decay factor for Exponential Moving Average of model parameters")
    
    parser.add_argument("--by-iteration", action="store_true", help="convert scheduler to be per iteration, not per epoch")
    parser.add_argument("--lr-scheduler", default="cosa", type=str, choices=["step", "cosa", "exp"], help="lr scheduler")
    parser.add_argument("--lr-step", default=20, type=int, help="period of learning rate decay")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="multiplicative factor of learning rate decay")
    parser.add_argument("--lr-warmup-epochs", default=5, type=int, help="number of epochs to warmup")
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="decay of learning rate in warmup stage")
    
    parser.add_argument("--amp", action="store_true", help="use automatic mixed precision")
    parser.add_argument("--workers", default=8, type=int, help="number of data loading workers")
    parser.add_argument("--model-name", default="sew_resnet18", type=str, help="name of model to train")
    parser.add_argument("--not-snn", action="store_true", help="model is not a snn")
    parser.add_argument('--T', default=4, type=int, help="total time-steps")
    parser.add_argument("--output-path", default="logs/", help="path to save outputs")
    parser.add_argument("--print-freq", default=1000, type=int, help="print frequency")
    
    parser.add_argument("--local_rank", default=0, type=int, help="node rank for distributed training")
    args = parser.parse_args()
    return args


def set_deterministic():
    random.seed(_seed_)
    np.random.seed(_seed_)
    torch.manual_seed(_seed_)
    torch.cuda.manual_seed_all(_seed_)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(_seed_)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    

def _get_cache_path(filepath, dataset_name):
    import hashlib
    
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("/userhome", "datasets", dataset_name, h[:10] + ".pt")
    return cache_path


def load_data(args, interpolation=InterpolationMode.BILINEAR):
    print("Loading data...")
    
    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(os.path.join(args.data_path, "train"), "ImageNet")
    if args.cache_dataset and os.path.exists(cache_path):
        print(f"Loading train dataset from {cache_path}")
        train_set, _ = torch.load(cache_path)
    else:
        transforms_train = [torchvision.transforms.RandomResizedCrop(args.train_crop_size, interpolation=interpolation)]
        transforms_train.append(torchvision.transforms.RandomHorizontalFlip(0.5))
        if args.auto_augment is not None:
            if args.auto_augment == "ta_wide":
                transforms_train.append(torchvision.transforms.TrivialAugmentWide(interpolation=interpolation))
            elif args.auto_augment == "ra":
                transforms_train.append(torchvision.transforms.RandAugment(interpolation=interpolation))
            else:
                aa_policy = torchvision.transforms.AutoAugmentPolicy(args.auto_augment)
                transforms_train.append(torchvision.transforms.AutoAugment(policy=aa_policy, interpolation=interpolation))
        transforms_train.extend([
            torchvision.transforms.PILToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.float),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        if args.random_erase > 0:
            transforms_train.append(torchvision.transforms.RandomErasing(p=args.random_erase))
        transforms_train = torchvision.transforms.Compose(transforms_train)
        
        train_set = torchvision.datasets.ImageFolder(root=os.path.join(args.data_path, "train"),
                                                     transform=transforms_train)
        if args.cache_dataset:
            print(f"Saving train dataset to {cache_path}")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            utils.save_on_master((train_set, os.path.join(args.data_path, "train")), cache_path)
    print(f"Length of training data: {len(train_set)}")
    print(f"Took {time.time() - st}s")
    
    print("Loading validation data")
    st = time.time()
    cache_path = _get_cache_path(os.path.join(args.data_path, "test"), "ImageNet")
    if args.cache_dataset and os.path.exists(cache_path):
        print(f"Loading test dataset from {cache_path}")
        test_set, _ = torch.load(cache_path)
    else:
        transforms_test = torchvision.transforms.Compose([
            torchvision.transforms.Resize(args.val_resize_size, interpolation=interpolation),
            torchvision.transforms.CenterCrop(args.val_crop_size),
            torchvision.transforms.PILToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.float),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        test_set = torchvision.datasets.ImageFolder(root=os.path.join(args.data_path, "val"),
                                                    transform=transforms_test)
        if args.cache_dataset:
            print(f"Saving test dataset to {cache_path}")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            utils.save_on_master((test_set, os.path.join(args.data_path, "test")), cache_path)
    print(f"Length of testing data: {len(test_set)}")
    print(f"Took {time.time() - st}s")
    
    print("Creating data loaders")
    g = torch.Generator()
    g.manual_seed(_seed_)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, seed=_seed_)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_set, generator=g)
        test_sampler = torch.utils.data.SequentialSampler(test_set)
    
    return train_set, test_set, train_sampler, test_sampler


def load_model(args):
    return eval(f"{args.model_name}")(num_classes=1000, cnf="ADD")


def get_logdir_name(args):
    logdir = f"{args.model_name}/" \
             f"T{args.T}_epochs{args.epochs}_bs{args.batch_size}_" \
             f"{args.opt}_lr{args.lr}_momentum{args.momentum}_wd{args.weight_decay}_" \
             f"aaug{args.auto_augment}_re{args.random_erase}_ma{args.mixup_alpha}_ca{args.cutmix_alpha}_" \
             f"size{args.train_crop_size}_{args.val_resize_size}_{args.val_crop_size}_"
    if args.lr_scheduler == "step":
        logdir += f"step_lrstep{args.lr_step}_gamma{args.lr_gamma}_"
    elif args.lr_scheduler == "cosa":
        logdir += f"cosa_"
    elif args.lr_scheduler == "exp":
        logdir += f"exp_gamma{args.lr_gamma}_"
    logdir += f"ws{args.world_size if args.distributed else 1}"
    return logdir


def set_optimizer(parameters, args):
    if args.opt == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == "adam":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer


def set_lr_scheduler(optimizer, iters_per_epoch, args):
    if args.lr_scheduler == "step":
        if args.by_iteration:
            lr_step = iters_per_epoch * args.lr_step
        else:
            lr_step = args.lr_step
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosa":
        if args.by_iteration:
            t_max = iters_per_epoch * (args.epochs - args.lr_warmup_epochs)
        else:
            t_max = args.epochs - args.lr_warmup_epochs
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    elif args.lr_scheduler == "exp":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    
    if args.lr_warmup_epochs > 0:
        if args.by_iteration:
            warmup_iters = iters_per_epoch * args.lr_warmup_epochs
        else:
            warmup_iters = args.lr_warmup_epochs
        warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=args.lr_warmup_decay, total_iters=warmup_iters)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[warmup_iters])
    else:
        lr_scheduler = main_lr_scheduler
    return lr_scheduler


def train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, args, model_ema=None, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("loss", utils.SmoothedValue(window_size=20, fmt="{global_avg:.4f}"))
    metric_logger.add_meter("acc", utils.SmoothedValue(window_size=20, fmt="{global_avg:.4f}"))
#     metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))
    
    header = f"Epoch: [{epoch}]"
    for i, (inputs, labels) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        inputs, labels = inputs.to(device), labels.to(device)
        if not args.not_snn:
            inputs = inputs.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(inputs)
            if not args.not_snn:
                outputs = outputs.mean(0)
            loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        if not args.not_snn:
            functional.reset_net(model)
        
        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)
        
        acc = utils.accuracy(outputs, labels)
        batch_size = labels.size(0)
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc"].update(acc.item(), n=batch_size)
#         metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
        if args.by_iteration:
            lr_scheduler.step()
    
    metric_logger.synchronize_between_processes()
    train_loss, train_acc = metric_logger.meters["loss"].global_avg, metric_logger.meters["acc"].global_avg
    
    return train_loss, train_acc


def evaluate(model, criterion, data_loader, device, args, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("loss", utils.SmoothedValue(window_size=20, fmt="{global_avg:.4f}"))
    metric_logger.add_meter("acc", utils.SmoothedValue(window_size=20, fmt="{global_avg:.4f}"))
#     metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))
    
    header = f"Test: {log_suffix}"
    num_processed_samples = 0
    
    with torch.inference_mode():
        for i, (inputs, labels) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
            start_time = time.time()
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if not args.not_snn:
                inputs = inputs.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)
            
            outputs = model(inputs)
            if not args.not_snn:
                outputs = outputs.mean(0)
            loss = criterion(outputs, labels)
            
            if not args.not_snn:
                functional.reset_net(model)
            
            acc = utils.accuracy(outputs, labels)
            batch_size = labels.size(0)
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc"].update(acc.item(), n=batch_size)
#             metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

            # FIXME need to take into account that the datasets could have been padded in distributed setup
            num_processed_samples += batch_size
            
    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
         warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )
    
    metric_logger.synchronize_between_processes()
    test_loss, test_acc = metric_logger.meters["loss"].global_avg, metric_logger.meters["acc"].global_avg
    
    return test_loss, test_acc


def main(args):
    set_deterministic()
    utils.init_distributed_mode(args)
    print(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_set, test_set, train_sampler, test_sampler = load_data(args)
    
    collate_fn = None
    num_classes = len(train_set.classes)
    mixup_transforms = []
    if args.mixup_alpha > 0:
        mixup_transforms.append(transforms.RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha))
    if args.cutmix_alpha > 0:
        mixup_transforms.append(transforms.RandomCutmix(num_classes, p=1.0, alpha=args.cutmix_alpha))
    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
        collate_fn = lambda batch: mixupcutmix(*default_collate(batch))
    
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn = collate_fn,
        worker_init_fn=seed_worker
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, 
        batch_size=args.batch_size,
        sampler=test_sampler, 
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=seed_worker
    )
    
    print("Creating model")
    model = load_model(args)
    if not args.not_snn:
        functional.set_step_mode(model, 'm')
        functional.set_backend(model, 'cupy', neuron.BaseNode)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = set_optimizer(model.parameters(), args)
    lr_scheduler = set_lr_scheduler(optimizer, len(train_loader), args)
    
    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent from other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and ommit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)
    
    if utils.is_main_process():
        logdir = get_logdir_name(args)
        logdir = os.path.join(args.output_path, logdir)
        os.makedirs(logdir, exist_ok=True)
        
        writer = SummaryWriter(logdir)
        with open(os.path.join(logdir, "args.txt"), 'w') as f:
            f.write(str(args))
        
        max_test_acc = -1.
        if model_ema:
            max_ema_test_acc = -1.
    
    print("Start training...")
    st = time.time()
    for epoch in range(args.epochs):
        start_time = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        train_loss, train_accuracy = train_one_epoch(model, criterion, optimizer, lr_scheduler, train_loader, device, epoch, args, model_ema, scaler=scaler)
        if not args.by_iteration:
            lr_scheduler.step()
        test_loss, test_accuracy = evaluate(model, criterion, test_loader, device, args)
        
        if model_ema:
            ema_test_loss, ema_test_accuracy = evaluate(model_ema, criterion, test_loader, device, args, log_suffix="EMA")
        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        
        if utils.is_main_process():
            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("train_accuracy", train_accuracy, epoch)
            writer.add_scalar("test_loss", test_loss, epoch)
            writer.add_scalar("test_accuracy", test_accuracy, epoch)
            if model_ema:
                writer.add_scalar("ema_test_loss", ema_test_loss, epoch)
                writer.add_scalar("ema_test_accuracy", ema_test_accuracy, epoch)
            
            save_max_test_acc = False
            save_max_ema_test_acc = False
            
            if test_accuracy > max_test_acc:
                max_test_acc = test_accuracy
                save_max_test_acc = True
            
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
                "test_acc": test_accuracy,
                "max_test_acc": max_test_acc,
            }
            if scaler is not None:
                checkpoint["scaler"] = scaler.state_dict()
                
            if model_ema:
                if ema_test_accuracy > max_ema_test_acc:
                    max_ema_test_acc = ema_test_accuracy
                    save_max_ema_test_acc = True
                checkpoint["model_ema"] = model_ema.state_dict()
                checkpoint["ema_test_acc"] = ema_test_accuracy
                checkpoint["max_ema_test_acc"] = max_ema_test_acc
            
            utils.save_on_master(checkpoint, os.path.join(logdir, f"checkpoint_latest.pth"))
            if save_max_test_acc:
                utils.save_on_master(checkpoint, os.path.join(logdir, f"checkpoint_max_test_acc.pth"))
            if model_ema and save_max_ema_test_acc:
                utils.save_on_master(checkpoint, os.path.join(logdir, f"checkpoint_max_ema_test_acc.pth"))
        
        print_results = f"Total: train_acc={train_accuracy:.4f}  train_loss={train_loss:.4f}  test_acc={test_accuracy:.4f}  test_loss={test_loss:.4f}  "
        if model_ema:
            print_results += f"ema_test_acc={ema_test_accuracy:.4f}  ema_test_loss={ema_test_loss:.4f}  "
        print_results += f"total time={total_time_str}"
        print(print_results)
        print()
    
    total_time = time.time() - st
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Took {total_time_str}")


if __name__ == "__main__":
    args = get_args()
    main(args)
