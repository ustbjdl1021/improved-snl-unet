# Segmentation and Measurement of Superalloy Microstructure Based on Improved Nonlocal Block

This repo is the official implementation for [Segmentation and Measurement of Superalloy Microstructure Based on Improved Nonlocal Block](https://ieeexplore.ieee.org/document/9739731). 

## improved-snl-unet

![improved-snl-block](https://github.com/ustbjdl1021/improved-snl-unet/blob/main/pic/block.png)

## Requirements

- python 3.8

- pytorch 1.8.1

- torchvision 0.9.1

- albumentations==0.5.2

## Usage

### train

run the `run.sh` or run the train script. You can tune the hyperparameters yourself according to your needs.

```shell
python train.py --epochs 100 --batch-size 20 --learning-rate 0.00001 --scale 1.0 --validation 10.0
```

Also, if you want to use multi-gpu training, run the multi-gpu training script.

```shell
python -m torch.distributed.launch --nproc_per_node=2 --use_env train_multigpu.py --epochs 10 --batch-size 12 --learning-rate 0.00001 --scale 1.0 --validation 10.0
```

### predict

This paper uses the symmetric overlap-tile strategy to predict images of any size. For details, please refer to [Deep Learning-Based Image Segmentation for Al-La Alloy Microscopic Images](https://www.mdpi.com/2073-8994/10/4/107).

```shell
python predict_anysize.py --model checkpoint_epoch100.pth --input dataset/val/test
```

## Citation

If you use our code or model in your work or find it is helpful, please cite the paper:

  ```bib
  @ARTICLE{9739731,  
  author={Zhang, Lixin and Yao, Haotian and Xu, Zhengguang and Sun, Han},  journal={IEEE Access},   
  title={Segmentation and Measurement of Superalloy Microstructure Based on Improved Nonlocal Block},   
  year={2022},  
  volume={10},  
  number={},  
  pages={32418-32425},  
  doi={10.1109/ACCESS.2022.3161507}}
  ```

## reference

1. [SNL_ICCV2021](https://github.com/zh460045050/SNL_ICCV2021)
2. [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
3. [Deep Learning-Based Image Segmentation for Al-La Alloy Microscopic Images](https://www.mdpi.com/2073-8994/10/4/107)
