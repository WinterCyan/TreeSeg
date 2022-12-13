import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class WeightedTverskyLoss(nn.Module):
    def __init__(self, alpha=0.6, beta=0.4, smooth=1e-5, weight=None, size_average=True):
        super(WeightedTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta 
        self.smooth = smooth

    def forward(self, inputs, targets):
        """ 
        inputs is output before sigmoid (not probs)
        targets contains annotation & weight
        """
        print(f'tversky-inputs: {inputs.shape}, targets: {targets.shape}')

        assert targets.shape[1] == 2, "targets channel not 2"
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       

        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        labels = targets[:,0,:,:].unsqueeze(1)
        weights = targets[:,1,:,:].unsqueeze(1)
        print(f'lables: {labels.shape}, weights: {weights.shape}')
        labels = labels.contiguous().view(-1)
        weights = weights.contiguous().view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * labels * weights).sum()    
        FP = ((1-labels) * inputs * weights).sum()
        FN = (labels * (1-inputs) * weights).sum()

        tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)  
        
        return 1 - tversky

def accuracy(probs, label):
    return torch.eq(torch.round(label), torch.round(probs))

# TODO: ensure dim's correct
# def dice_coef(probs, label, smooth=0.0000001):
#     intersection = torch.sum(torch.abs(label * probs), axis=0)
#     union = torch.sum(label, axis=0) + torch.sum(probs, axis=0)
#     return torch.mean((2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    """
    inputs are masks
    """
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

# def dice_loss(probs, label):
#     return 1 - dice_coef(probs, label)

def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    return 1 - dice_coef(input, target, reduce_batch_first=True)

# calculate TP, FP, TN, FN: label is 0-1 annotation, probs is predicted probs, range[0,1]
def true_positives(probs, label):
    return torch.round(label * probs)

def false_positives(probs, label):
    return torch.round((1 - label) * probs)

def true_negatives(probs, label):
    return torch.round((1 - label) * (1 - probs))

def false_negatives(probs, label):
    return torch.round((label) * (1 - probs))

def sensitivity(probs, label):
    tp = true_positives(probs, label)
    fn = false_negatives(probs, label)
    return torch.sum(tp) / (torch.sum(tp) + torch.sum(fn))

def specificity(probs, label):
    tn = true_negatives(probs, label)
    fp = false_positives(probs, label)
    return torch.sum(tn) / (torch.sum(tn) + torch.sum(fp))
