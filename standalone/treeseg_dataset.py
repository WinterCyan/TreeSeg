from torch.utils.data import DataLoader, Dataset
from shutil import move as shmove
import logging
from random import shuffle
import os
from os import listdir
from os import path
from os.path import splitext
from os.path import join as pjoin
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from PIL.Image import open as Imgopen
from torch.utils.data import Dataset
import wandb
from tqdm import tqdm


def create_val_set(dataset_dir:str, set_len:int=1000, img_type=".png", thr=0.3):
    val_dir = f'{dataset_dir}/val'
    if os.path.exists(f'{val_dir}') and len(os.listdir(f'{val_dir}'))>0:
        print('val dataset exists, exit.')
        return
    if not path.exists(val_dir):
        os.makedirs(val_dir)
    valid_names = []
    suffix_names = [f.replace('pan', '') for f in listdir(dataset_dir) if f.startswith('pan-') and f.endswith(img_type)]
    for n in suffix_names:
        annotation_arr = np.asarray(Imgopen(pjoin(dataset_dir, f'annotation{n}')))
        if np.mean(annotation_arr) >= thr:
            valid_names.append(n)
    print(f'valid count: {len(valid_names)}')
    shuffle(valid_names)
    valset_names = valid_names[:set_len]
    types = ['pan','ndvi','annotation','boundary']
    with open(f'{val_dir}/val_names.txt', 'w') as f:
        for n in valset_names:
            for t in types:
                shmove(f'{dataset_dir}/{t}{n}', f'{val_dir}/{t}{n}')
                f.write(f'{t}{n}\n')
    f.close()
    print(f'selected {set_len} samples to val set')
    return valset_names


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        # resize dataset img
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        # shape as [channel, H, W]
        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def load(filename):
        # load img file
        ext = splitext(filename)[1]
        if ext == '.npy':
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Imgopen(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')


# TODO: data augmentation
"""
    1. flip, left-right, up-down (+2)
    2. rotation, +90, 180, -90 (+3)
    3. flip + rotation, (+2)
"""
class TreeDataset(Dataset):
    def __init__(self, dataset_dir:str, img_type='.png', mean_filter=True, annotation_thr=0.02):
        # dataset_dir contains [pan, ndvi, boundary, annotation] imgs
        super(TreeDataset).__init__()
        self.dataset_shape = (256,256)
        self.thr = annotation_thr

        self.dataset_dir = Path(dataset_dir)
        suffix_names = [f.replace('pan', '') for f in listdir(dataset_dir) if f.startswith('pan-') and f.endswith(img_type)]
        self.valid_names = self.mean_filter_out(suffix_names) if mean_filter else suffix_names

        if not (os.path.exists(f'{dataset_dir}/val') and len(os.listdir(f'{dataset_dir}/val'))>0):
            if not dataset_dir.endswith('val'):
                valset_names = create_val_set(dataset_dir)
                for n in valset_names:
                    self.valid_names.remove(n)
                print('val dataset created.')

    def __len__(self):
        return len(self.valid_names)

    def mean_filter_out(self, names):
        valid_names = []
        for n in names:
            annotation_arr = np.asarray(Imgopen(pjoin(self.dataset_dir, f'annotation{n}')))
            if np.mean(annotation_arr) >= self.thr:
                valid_names.append(n)
        return valid_names

    @staticmethod
    def preprocess(pil_img, resize_shape, is_mask, polarize=False):
        pil_img = pil_img.resize(resize_shape, resample=Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR)
        img_ndarray = np.array(pil_img)

        # transpose for 2-channel or 3-channel img
        if img_ndarray.ndim == 2:
            img_ndarray = img_ndarray[np.newaxis, ...]
        else:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        # polarize boundary
        if polarize:
            img_ndarray[img_ndarray>=0.5] = 10
            img_ndarray[img_ndarray<0.5] = 1

        return img_ndarray

    def __getitem__(self, index):
        name = self.valid_names[index]
        pan = self.preprocess(Imgopen(pjoin(self.dataset_dir, f"pan{name}")), self.dataset_shape, is_mask=False, polarize=False)
        ndvi = self.preprocess(Imgopen(pjoin(self.dataset_dir, f"ndvi{name}")), self.dataset_shape, is_mask=False, polarize=False)
        annotation = self.preprocess(Imgopen(pjoin(self.dataset_dir, f"annotation{name}")), self.dataset_shape, is_mask=True, polarize=False)
        boundary = self.preprocess(Imgopen(pjoin(self.dataset_dir, f"boundary{name}")), self.dataset_shape, is_mask=True, polarize=True)

        return {
            # is_contiguous直观的解释是Tensor底层一维数组元素的存储顺序与Tensor按行优先一维展开的元素顺序是否一致。
            'pan': torch.as_tensor(pan.copy()),
            'ndvi': torch.as_tensor(ndvi.copy()),
            'annotation': torch.as_tensor(annotation.copy()),
            'boundary': torch.as_tensor(boundary.copy())
        }

if __name__ == '__main__':
    # wandb.init()

    dir = "/home/winter/code-resource/treeseg/trainingdata/trainsample_128_onsample"
    # dir = "/home/lenovo/treeseg-dataset/full_process/sample_128_nonorm"
    # dataset = TreeDataset(dir, annotation_thr=0)
    # print(len(dataset))
    # loader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=4, pin_memory=False)
    # print(len(loader))
    create_val_set(dir, thr=0.3)