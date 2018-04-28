import os
import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt

from torch.utils import data

from ptsemseg.utils import recursive_glob

class MPIBlurLoader(data.Dataset):
    def __init__(self, root, split="training", is_transform=False, img_size=(6000,4000), augmentations=None, img_norm=True):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 2
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)

        for split in ["training", "validation",]:
            file_list = recursive_glob(rootdir=self.root + 'images/' + self.split + '/', suffix='.jpg')
            self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
        lbl_path = img_path[:-4] + '_seg.png'

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = m.imread(lbl_path)
        lbl = np.array(lbl, dtype=np.int32)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl


    def transform(self, img, lbl):
        img = m.imresize(img, (self.img_size[0], self.img_size[1])) # uint8 with RGB mode
        img = img[:, :, ::-1] # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

		
        lbl = lbl.astype(int)
        lbl = torch.from_numpy(lbl).long()
        return img, lbl




if __name__ == '__main__':
    local_path = 'C:\data\Synthetic_blur_MPI_data\images'
    dst = ADE20KLoader(local_path, is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
