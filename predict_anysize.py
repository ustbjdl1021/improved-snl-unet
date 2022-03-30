import argparse
import logging
import os
import glob
import xlwt
import time

import numpy as np
import torch
from torch import cuda
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms
from utils.overlap_tile_crop_cat import cropfullimg, concat_img_patchs

from utils.data_loading import BasicDataset
from model.snl_unet import SNLUNet
from utils.metrics import Evaluator
from utils.metrics_more import *


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy(), output


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='InputFloderName', required=True)
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1,
                        help='Scale factor for the input images')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255).astype(np.uint8))


if __name__ == '__main__':
    args = get_args()
    fileList = []
    for root, dirs, files in os.walk(args.input[0]):
        for file in files:
            fileList.append(os.path.join(root, file))
    in_files = fileList

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = SNLUNet(n_channels=3, n_classes=2, bilinear=True)

    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')

    # record learnable parameters
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("total_params:{}".format(total_params))

    evaluator = Evaluator(2)
    evaluator.reset()

    # create excel
    workbook = xlwt.Workbook(encoding= 'utf-8')
    # Create a new sheet
    worksheet = workbook.add_sheet("new Sheet01")
    worksheet.write(0, 1, "Vf")
    worksheet.write(0, 2, "omega")
    worksheet.write(0, 3, "D")

    # record time used
    time_start = time.time()

    metric_names = ["pixel_accuracy", "mean_accuracy",
                "iou", "fwiou", "dice",
                "ri", "ari", "me", "se", "vi",
                "cardinality_difference", "map"]
    seg_total_eval = np.zeros((len(metric_names), len(in_files)))

    for i, fileDir in enumerate(in_files):
        logging.info(f'\nPredicting image {fileDir} ...')
        labelDir = fileDir.replace("test", "label")
        img = Image.open(fileDir).convert("RGB")
        img = ImageOps.equalize(img, mask=None)  # Histogram equalization
        label = Image.open(labelDir)
        label = torch.from_numpy(BasicDataset.preprocess(label, scale=1.0, is_mask=True))
        label = label.unsqueeze(0)

        img_patchs, center_crops, big_patch_crops = cropfullimg(img)
        mask_patchs =[]
        output_patchs =[]
        for img_patch in img_patchs:
            mask, output = predict_img(net=net,
                                       full_img=img_patch,
                                       scale_factor=args.scale,
                                       out_threshold=args.mask_threshold,
                                       device=device)
            mask = mask_to_image(mask)
            mask_patchs.append(mask)
            output_patchs.append(mask_patchs)
        mask = concat_img_patchs(mask_patchs, center_crops, img.size)
        output = concat_img_patchs(output_patchs, center_crops, img.size)


        filename = fileDir.split("/")[-1].split(".")[0]
        # out_filename = out_files[i]
        out_filename = fileDir.split('.')[0] + '_snlunet_res.tif'
        # outimg = mask_to_image(mask)

        # calculate miou
        result = output.data.cpu().numpy()
        result = np.argmax(result, axis=1)
        label = label.float().cpu().numpy()
        # Add batch sample into evaluator
        evaluator.add_batch(label, result)

        # more metrics(from maboyuan)
        metric_values = get_total_evaluation(result, label, require_edge=False)
        for metric_idx, metric_value in enumerate(metric_values):
            seg_total_eval[metric_idx, i] = metric_value

        # Calculate Volume fraction
        pixels = result.getdata()
        brightTotal = len(list(filter(lambda i: i >= 250, pixels)))
        volume_fraction = brightTotal / (img.size[0] * img.size[1])

        # Calculate perimeter fraction
        edges_img = result.filter(ImageFilter.FIND_EDGES)
        edges_pixels = edges_img.getdata()
        perimeter = len(list(filter(lambda i: i >= 250, edges_pixels)))
        perimeter_fraction = perimeter / (img.size[0] * img.size[1])

        # write into excel
        worksheet.write(i+1, 0, filename)
        worksheet.write(i+1, 1, volume_fraction)
        worksheet.write(i+1, 2, perimeter_fraction)

        # result = outputimg
        mask.save(out_filename)
        logging.info(f'Mask saved to {out_filename}')

    time_end = time.time()
    average_time = (time_end - time_start) / len(in_files)
    print("predict {} seconds per img".format(average_time))
    # Acc = evaluator.Pixel_Accuracy()
    # mIoU = evaluator.Mean_Intersection_over_Union()
    # print('Acc: {} '.format(Acc))
    # print('mIoU: {} '.format(mIoU))
    workbook.save("Measurement.xls")

