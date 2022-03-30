#!/bin/bash

module load anaconda/2020.11
module load cuda/10.2
module load cudnn/7.6.5.32_cuda10.2
module load tensorboard/2.3.0
source activate torch
# python train.py --epochs 30 --batch-size 12 --learning-rate 0.00001 --scale 1.0 --validation 10.0
# python predict.py --model checkpoints/checkpoint_epoch30.pth --input dataset/val/test/1000-44-1200-z.tif --output dataset/val/test/1000-44-1200-z_res.tif
python train.py --epochs 100 --batch-size 20 --learning-rate 0.00001 --scale 1.0 --validation 10.0
# python predict.py --model checkpoints/unet/checkpoint_epoch100.pth --input 235-A1-H-GAN-WY-20K-050.tif --output 235-A1-H-GAN-WY-20K-050_res.tif
# python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py --epochs 10 --batch-size 12 --learning-rate 0.00001 --scale 1.0 --validation 10.0
