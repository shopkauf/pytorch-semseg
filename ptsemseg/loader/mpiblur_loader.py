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
    def __init__(self, root, split="training", is_transform=True, img_size=(300,300), augmentations=None, img_norm=True):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 2
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        #self.files = collections.defaultdict(list)

        #for split in ["training", "validation",]:
            #file_list = recursive_glob(rootdir=self.root + 'images/' + self.split + '/', suffix='.png')
        file_list = recursive_glob(r'C:\data\Synthetic_blur_MPI_data\images\\' + self.split + '/', suffix='.png')
            #file_list = recursive_glob(r'C:\data\Synthetic_blur_MPI_data\images\\' + split + '/', suffix='.png')
            #self.files[split] = file_list
        self.files = file_list
        print(file_list)
			
		
    def __len__(self):
        #return len(self.files[self.split])
        return len(self.files)

    def __getitem__(self, index):
        #img_path = self.files[self.split][index].rstrip() ## removes white spaces at the end
        img_path = self.files[index].rstrip() ## removes white spaces at the end
        folder, filename = os.path.split(img_path) ## folder name is 'training' or 'validation'
        parent_folder, child_folder = os.path.split(folder)
        #print("parent_folder is ", parent_folder)
		
        lbl_path = os.path.join(parent_folder, 'labels', filename[:-4] + '_seg.png')

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
        #print("img size is ", img.shape[0], img.shape[1])
        #print("lbl size is ", lbl.shape[0], lbl.shape[1])
        #img = m.imresize(img, (self.img_size[0], self.img_size[1])) # uint8 with RGB mode
        #print("now, img size is ", img.shape[0], img.shape[1])
        tmp = np.array(img)
        #print("before transform, tmp max is ", tmp[:,:,0].max(), tmp[:,:,0].min())
        img = img[:, :, ::-1] # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        #tmp = np.array(img)
        #print("after reduce mean, tmp max is ", tmp[:,:,0].max(), tmp[:,:,0].min())
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        #tmp = np.array(img)
        #print("after norm, tmp max is ", tmp[:,:,0].max(), tmp[:,:,0].min())	
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

		
        lbl = lbl.astype(int)
        #tmp = np.array(lbl)
        #print("before transform, lbl max and min is ", lbl[:,:].max(), lbl[:,:].min())		
        #lbl = torch.from_numpy(lbl).long()
        lbl = np.expand_dims(lbl, 0) ## change dimension: [224,224] become [1,224,224]
        lbl = torch.from_numpy(lbl).float()
        return img, lbl




if __name__ == '__main__':
    local_path = 'C:\data\Synthetic_blur_MPI_data\images'

    dst = MPIBlurLoader(local_path, is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)

