import argparse
import logging
from re import A
import sys
from pathlib import Path
import os
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
# import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utils.data_loading import BasicDataset, MyDataset
from utils.dice_score import dice_loss
from evaluate_multigpus import evaluate
from model.snl_unet import SNLUNet
from utils.metrics import Evaluator
from multi_train_utils.distributed_utils import init_distributed_mode, dist, cleanup, reduce_value, is_main_process

dir_img = Path('dataset/origindataset/train/')
dir_mask = Path('dataset/origindataset/label/')
dir_checkpoint = Path('checkpoints/SNLUNet/')


def train_net(net,
              device,
              rank,
              weights_path,
              epochs: int = 10,
              batch_size: int = 12,
              learning_rate: float = 0.00001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 1,
              amp: bool = False):
    # Create dataset
    try:
        dataset = MyDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # tensorboard
    if rank == 0:
        tb_writer = SummaryWriter(log_dir="runs/test")
    
    # Assign the training sample index to the process corresponding to each rank
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)

    # Form a list of sample indices every batch_size elements
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)
    # number of workers
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    # 3. Create data loaders
    loader_args = dict(num_workers=0, pin_memory=False)
    train_loader = DataLoader(train_set, batch_sampler=train_batch_sampler, **loader_args)
    val_loader = DataLoader(val_set, batch_size = batch_size, sampler=val_sampler, **loader_args)

    # Load pretrained weights if they exist
    if os.path.exists(weights_path):
        weights_dict = torch.load(weights_path, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if net.state_dict()[k].numel() == v.numel()}
        net.load_state_dict(load_weights_dict, strict=False)
    else:
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
        '''
        If there is no pre-training weight, you need to save the weight in the first process, 
        and then load it in other processes to keep the initialization weight consistent
        '''
        if rank == 0:
            torch.save(net.state_dict(), checkpoint_path)

        dist.barrier()
        # Note here that the map_location parameter must be specified, otherwise the first GPU will take up more resources
        net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # It only makes sense to use SyncBatchNorm when training a network with a BN structure
    if args.syncBN:
        # Training takes longer after using SyncBatchNorm
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).to(device)

    # Convert to DDP model
    DDPmodel = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], find_unused_parameters=True)

    if rank == 0:
        logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_checkpoint}
            Device:          {device.type}
            Images scaling:  {img_scale}
            Mixed Precision: {amp}
        ''')

    # Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(DDPmodel.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0
    evaluator = Evaluator(2)
    evaluator.reset()
    
    # Begin training
    for epoch in range(epochs):
        DDPmodel.train()
        epoch_loss = 0

        if rank == 0:
            pbar = tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img')
 
        for batch in train_loader:
            images = batch['image']
            true_masks = batch['mask']

            assert images.shape[1] == net.n_channels, \
                f'Network has been defined with {net.n_channels} input channels, ' \
                f'but loaded images have {images.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'

            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            with torch.cuda.amp.autocast(enabled=amp):
                masks_pred = DDPmodel(images)
                loss = criterion(masks_pred, true_masks) \
                        + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                    F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                    multiclass=True)

            # acc
            output = masks_pred.data.cpu().numpy()
            label = true_masks.float().cpu().numpy()
            output = np.argmax(output, axis=1)
            # Add batch sample into evaluator
            evaluator.add_batch(label, output)

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            if is_main_process():
                pbar.update(images.shape[0])
            global_step += 1
            epoch_loss += loss.item()
            if is_main_process():
                pbar.set_postfix(**{'loss (batch)': loss.item()})

            # Evaluation round
            if global_step % (n_train // batch_size) == 0:
                val_score = evaluate(DDPmodel, net, val_loader, device)
                scheduler.step(val_score)
                if is_main_process():
                    logging.info('Validation Dice score: {}'.format(val_score))

        # Wait for all processes to finish calculating
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        if is_main_process():
            Acc = evaluator.Pixel_Accuracy()
            mIoU = evaluator.Mean_Intersection_over_Union()
            logging.info('epoch{} Acc: {} '.format(epoch + 1, Acc))
            logging.info('epoch{} mIoU: {} '.format(epoch + 1, mIoU))
            tags = ["loss", "accuracy", "mIoU", "learning_rate"]
            tb_writer.add_scalar(tags[0], epoch_loss, epoch)
            tb_writer.add_scalar(tags[1], Acc, epoch)
            tb_writer.add_scalar(tags[2], mIoU, epoch)
            tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)

        if rank == 0:
            if save_checkpoint:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                torch.save(DDPmodel.module.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
                logging.info(f'Checkpoint {epoch + 1} saved!')
    
    # delete temporary cache files
    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)
    cleanup()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--syncBN', type=bool, default=True, help='Whether to enable SyncBatchNorm')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    # Do not change this parameter, the system will automatically assign
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # The number of open processes, do not set this parameter, it will be automatically set according to nproc_per_node
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # Initialize each process environment
    init_distributed_mode(args=args)
    rank = args.rank
    device = torch.device(args.device)
    batch_size = args.batch_size
    num_classes = args.num_classes
    weights_path = args.weights
    args.lr *= args.world_size  # The learning rate is multiplied by the number of parallel GPUs

    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = SNLUNet(n_channels=3, n_classes=2, bilinear=True)

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        if rank == 0:
            logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp,
                  rank = args.rank,
                  weights_path = args.weights)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
