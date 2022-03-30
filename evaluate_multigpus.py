import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff
from multi_train_utils.distributed_utils import is_main_process


def evaluate(DDPmodel, net, dataloader, device):
    DDPmodel.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # Print verification progress in main process
    if is_main_process():
        dataloader = tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False)
    # iterate over the validation set
    for batch in dataloader:
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = DDPmodel(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

    # Wait for all processes to finish calculating
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)       

    DDPmodel.train()
    return dice_score / num_val_batches
