from torch.utils.data import DataLoader, Dataset

import logging
from os import listdir
from os.path import splitext
from os.path import join as pjoin
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


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
            return Image.open(filename)

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


class TreeDataset(Dataset):
    def __init__(self, dataset_dir, img_type='.png'):
        super(TreeDataset).__init__()
        self.dataset_shape = (256,256)

        self.dataset_dir = Path(dataset_dir)
        self.suffix_names = [f.replace('pan', '') for f in listdir(dataset_dir) if f.startswith('pan-') and f.endswith(img_type)]

    def __len__(self):
        return len(self.suffix_names)

    @staticmethod
    def preprocess(pil_img, resize_shape, is_mask=False):
        assert resize_shape.isinstance(tuple) and len(resize_shape)==2
        pil_img = pil_img.resize(resize_shape, resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)
        print(f"img arr: \n{img_ndarray}")

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255
        
        print(f"returned arr: \n{img_ndarray}")
        return img_ndarray

    def __getitem__(self, index):
        name = self.suffix_names[index]
        pan = self.preprocess(Image.open(pjoin(self.dataset_dir, f"pan{name}")), self.dataset_shape, False)
        ndvi = self.preprocess(Image.open(pjoin(self.dataset_dir, f"ndvi{name}")), self.dataset_shape, False)
        annotation = self.preprocess(Image.open(pjoin(self.dataset_dir, f"annotation{name}")), self.dataset_shape, True)
        boundary = self.preprocess(Image.open(pjoin(self.dataset_dir, f"boundary{name}")), self.dataset_shape, True)

        return {
            'pan': torch.as_tensor(pan.copy()).float().contiguous(),
            'ndvi': torch.as_tensor(ndvi.copy()).float().contiguous(),
            'annotation': torch.as_tensor(annotation.copy()).long().contiguous(),
            'boundary': torch.as_tensor(boundary.copy()).long().contiguous()
        }