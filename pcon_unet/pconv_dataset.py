import random
import torch
from PIL import Image
from glob import glob


class PCONVDataset(torch.utils.data.Dataset):
    def __init__(self, impaths, maskpaths, img_transform, mask_transform):
        super().__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        # use about 8M images in the challenge dataset
        self.paths = impaths
        self.mask_paths = maskpaths
        self.N_mask = len(self.mask_paths)

    def __getitem__(self, index):
        gt_img = Image.open(self.paths[index])
        gt_img = self.img_transform(gt_img.convert('RGB'))

        mask = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
        mask = self.mask_transform(mask.convert('RGB'))
        return gt_img * mask, mask, gt_img

    def __len__(self):
        return len(self.paths)