from torch.utils.data import DataLoader, Dataset

import logging
from os import listdir
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


class TreeDataset(Dataset):
    def __init__(self, dataset_dir, img_type='.png', annotation_thr=0.02):
        # dataset_dir contains [pan, ndvi, boundary, annotation] imgs
        super(TreeDataset).__init__()
        self.dataset_shape = (256,256)
        self.thr = annotation_thr

        self.dataset_dir = Path(dataset_dir)
        suffix_names = [f.replace('pan', '') for f in listdir(dataset_dir) if f.startswith('pan-') and f.endswith(img_type)]
        self.valid_names = self.mean_filter_out(suffix_names)

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
    wandb.init()

    dir = "/home/lenovo/treeseg-dataset/full_process/temp"
    # dir = "/home/lenovo/treeseg-dataset/full_process/sample_128_nonorm"
    dataset = TreeDataset(dir)
    print(len(dataset))
    loader = DataLoader(dataset, shuffle=True, batch_size=4, num_workers=4, pin_memory=False)
    for i,batch in enumerate(tqdm(loader)):
        # print(f'pan min-max: {torch.min(batch["pan"])}, {torch.max(batch["pan"])}')
        # print(f'ndvi min-max: {torch.min(batch["ndvi"])}, {torch.max(batch["ndvi"])}')
        # print(f'anno unique: {np.unique(batch["annotation"])}')
        # print(f'bound unique: {np.unique(batch["boundary"])}')

        pan_batch = batch['pan']
        ndvi_batch = batch['ndvi']
        annotation_batch = batch['annotation']
        boundary_batch = batch['boundary']
        print(f'pan ndvi anno boundary | devices: {pan_batch.device},{ndvi_batch.device},{annotation_batch.device},{boundary_batch.device}')

        log_img_pan = wandb.Image(batch['pan'], caption='pan')
        log_img_ndvi = wandb.Image(batch['ndvi'], caption='ndvi')
        log_img_annotation = wandb.Image(batch['annotation'], caption='annotation')
        log_img_boundary = wandb.Image(batch['boundary'], caption='boundary')

        wandb.log({
            'pan': log_img_pan,
            'ndvi': log_img_ndvi,
            'annotation': log_img_annotation,
            'boundary': log_img_boundary
        })
