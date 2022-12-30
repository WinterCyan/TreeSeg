import argparse
import logging
from pathlib import Path
import os
from os import path
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
# ----------------------------------------
# 因为默认的file_descriptor共享策略使用文件描述符作为共享内存句柄，并且当DataLoader上有太多批次时，这将达到限制。
# 要解决此问题，您可以通过将其添加到脚本来切换到file_system策略。
from torch import multiprocessing
multiprocessing.set_sharing_strategy('file_system')
# ----------------------------------------
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from treeseg_dataset import TreeDataset, BasicDataset
from unet_repo import dice_loss, WeightedTverskyLoss
from torch_eval import evaluate
from unet_repo import UNet

def train_net(
        net,
        device,
        dir_dataset:str,
        dir_model:str,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        model_name: str = 'defaultname',
        amp: bool = False,
    ):

    dir_model = dir_model.rstrip('/')

    # 1. Create dataset
    train_dataset = TreeDataset(dir_dataset, annotation_thr=0)
    val_dataset = TreeDataset(f'{dir_dataset}/val', mean_filter=False)
    n_train = len(train_dataset)
    n_val = len(val_dataset)
    print(f"train dataset len: {n_train} [pan,ndvi,annotation,boundary] img pairs.")
    print(f"val dataset len: {n_val} [pan,ndvi,annotation,boundary] img pairs.")

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=8, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)
    print(f'train batch num: {len(train_loader)}')

    # (Initialize logging)
    experiment = wandb.init(project='TreeSeg, norm on sample', resume='allow', anonymous='must')
    experiment.config.update(
        dict(
            epochs=epochs, 
            batch_size=batch_size, 
            learning_rate=learning_rate,
            val_percent=val_percent, 
            save_checkpoint=save_checkpoint, 
            # img_scale=img_scale,
            amp=amp
        )
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adadelta(params=net.parameters(), lr=1.0, rho=0.95, eps=1e-6, weight_decay=0)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # criterion = nn.CrossEntropyLoss()
    criterion = WeightedTverskyLoss()
    global_step = 0

    log_freq_epoch = 3
    save_freq_epoch = 20
    division_step = (n_train // (log_freq_epoch * batch_size))
    if division_step == 0:
        division_step = 1

    # 5. Begin training
    for epoch in range(1, epochs+1):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                # images = batch['image']
                # true_masks = batch['mask']
                pan_batch = batch['pan']
                ndvi_batch = batch['ndvi']
                annotation_batch = batch['annotation']
                boundary_batch = batch['boundary']

                pan_batch = pan_batch.to(device=device, dtype=torch.float32)
                ndvi_batch = ndvi_batch.to(device=device, dtype=torch.float32)
                annotation_batch = annotation_batch.to(device=device, dtype=torch.float32)
                boundary_batch = boundary_batch.to(device=device, dtype=torch.float32)
                input_image = torch.concat((pan_batch, ndvi_batch), dim=1)
                target_tensor = torch.concat((annotation_batch, boundary_batch), dim=1)
                assert input_image.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {input_image.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                with torch.cuda.amp.autocast(enabled=amp):
                    direct_output = net(input_image)
                    loss = criterion(direct_output, target_tensor)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(input_image.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                if global_step % division_step == 0:
                    histograms = {}
                    for tag, value in net.named_parameters():
                        tag = tag.replace('/', '.')
                        histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                    eval_result = evaluate(net, val_loader, device, amp)
                    dice_score = eval_result['dice_score']
                    dice_loss = eval_result['dice_loss']
                    sensitivity = eval_result['sensitivity']
                    specificity = eval_result['specificity']
                    accuracy = eval_result['accuracy']
                    # scheduler.step(val_score)

                    log_img = torch.concat((pan_batch, ndvi_batch, annotation_batch.float(), direct_output.float()),dim=2)

                    logging.info(f'dice score: {dice_score}, dice loss: {dice_loss}, sensitivity: {sensitivity}, specificity: {specificity}, accuracy: {accuracy}')
                    experiment.log({
                        'dice_score': dice_score,
                        'dice_loss': dice_loss,
                        'sensitivity': sensitivity,
                        'specificity': specificity,
                        'accuracy': accuracy,
                        'learning rate': optimizer.param_groups[0]['lr'],
                        # 'pan images': wandb.Image(pan_batch.cpu()),
                        # 'ndvi images': wandb.Image(ndvi_batch.cpu()),
                        # 'masks': {
                        #     'true': wandb.Image(annotation_batch.float().cpu()),
                        #     'pred': wandb.Image(direct_output.float().cpu()),
                        # },
                        'input & output': wandb.Image(log_img.cpu()),
                        'step': global_step,
                        'epoch': epoch,
                        **histograms
                    })

        if save_checkpoint and epoch%save_freq_epoch==0:
            if not path.exists(dir_model):
                os.makedirs(dir_model)
            torch.save(net.state_dict(), f'{dir_model}/{model_name}_checkpoint_epoch{epoch}.pth')
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=5.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--model_name', type=str, default='defaultname')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=2, n_classes=args.classes, bilinear=args.bilinear)

    logging.info(
        f'Network:\n'
        f'\t{net.n_channels} input channels\n'
        f'\t{net.n_classes} output channels (classes)\n'
        f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling'
    )

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(
            net=net,
            epochs=args.epochs,
            dir_dataset=args.dataset_dir,
            dir_model=args.model_dir,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            val_percent=args.val / 100,
            model_name=args.model_name,
            amp=args.amp
        )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
