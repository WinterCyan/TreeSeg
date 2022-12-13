import torch
import torch.nn.functional as F
from tqdm import tqdm

from unet_repo import dice_loss, dice_coef, sensitivity, specificity, accuracy


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    total_dice_score = 0
    total_dice_loss = 0
    total_specificity = 0
    total_sensitivity = 0
    total_accuracy = 0

    # TODO: leverage weight-map in eval

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            pan, ndvi, annotation, boundary = batch['pan'], batch['ndvi'], batch['annotation'], batch['boundary']

            # move images and labels to correct device and type
            pan = pan.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            ndvi = ndvi.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            annotation = annotation.to(device=device, dtype=torch.float32)
            boundary = boundary.to(device=device, dtype=torch.float32)

            inp_concat = torch.concat((pan,ndvi), dim=1)

            # predict the mask
            output = net(inp_concat)
            probs_pred = torch.sigmoid(output)

            mask_pred = (probs_pred > 0.5).float()
            # compute the Dice score
            total_dice_score += dice_coef(mask_pred, annotation)
            total_dice_loss += dice_loss(mask_pred, annotation)
            total_sensitivity += sensitivity(probs_pred, annotation)
            total_specificity += specificity(probs_pred, annotation)
            total_accuracy += accuracy(probs_pred, annotation)

    net.train()
    return {
        'dice_score': total_dice_score / max(num_val_batches, 1),
        'dice_loss': total_dice_loss / max(num_val_batches, 1),
        'sensitivity': total_sensitivity / max(num_val_batches, 1),
        'specificity': total_specificity / max(num_val_batches, 1),
        'accuracy': total_accuracy / max(num_val_batches, 1),
    }