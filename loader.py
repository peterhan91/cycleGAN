import glob
import random
import os
import numpy as np
import torch
import scipy.ndimage
import nibabel as nib
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        # full path list of xrays 
        # self.files_A = sorted(glob.glob(os.path.join(root, '%s/xrays/frontal' % mode) + '/*.png'))[2:10000]
        # self.files_C = sorted(glob.glob(os.path.join(root, '%s/xrays/lateral' % mode) + '/*.png'))[2:10000]
        # full folder list of cts
        # self.files_B = sorted(glob.glob(os.path.join(root, '%s/cts/data' % mode) + '/*'))
        self.files_A = sorted(glob.glob(os.path.join(root, 'NIH_Chest/images', '*.png')))
        self.files_B = sorted(glob.glob(os.path.join(root, 'NLST/nii', '*.nii.gz')))

    def crop_center(self, vol, cropz, cropy, cropx):
        z,y,x = vol.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)
        startz = z//2-(cropz//2) 
        return vol[startz:startz+cropz, starty:starty+cropy, startx:startx+cropx]

    def normal_3D(self, vol):
        result = vol - vol.min()
        result = result / result.max() 
        return (result - 0.5) / 0.5

    def __getitem__(self, index):
        img_f = Image.open(self.files_A[index % len(self.files_A)]).convert('RGB')
        # img_l = Image.open(self.files_C[index % len(self.files_C)])
        img_f = self.transform(img_f)
        # img_l = self.transform(img_l)
        item_A = img_f
        # item_A = torch.stack((img_f.squeeze(0), img_l.squeeze(0)), dim=0)

        # item_B = np.float32(np.load(self.files_B[random.randint(0, len(self.files_B) - 1)] + '/128.npy'))
        try:
            item_B = nib.load(self.files_B[random.randint(0, len(self.files_B) - 1)]).get_fdata()
            dim_min = min(item_B.shape)
            while dim_min < 130:
                item_B = nib.load(self.files_B[random.randint(0, len(self.files_B) - 1)]).get_fdata()
                dim_min = min(item_B.shape)
            r = 128 / dim_min
            crop = self.crop_center(item_B, dim_min, dim_min, dim_min)
            crop = scipy.ndimage.interpolation.zoom(crop, r, mode='nearest')
            
            crop = self.normal_3D(crop)
            item_B = np.moveaxis(np.flipud(crop), 1, 0)
        except:
            return None

        return {'A': item_A, 
                'B': torch.from_numpy(np.expand_dims(item_B.copy(), axis=0)).type(torch.FloatTensor)}

    def __len__(self):
        return min(len(self.files_A), len(self.files_B))


class TestDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='test'):
        self.transform = transforms.Compose(transforms_)
        self.files_A = sorted(glob.glob(os.path.join(root, '%s/xrays/frontal' % mode) + '/*.png'))[:20]
        # self.files_C = sorted(glob.glob(os.path.join(root, '%s/xrays/lateral' % mode) + '/*.png'))[:20]

    def __getitem__(self, index):
        img_f = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        # img_l = self.transform(Image.open(self.files_C[index % len(self.files_C)]))
        item_A = img_f.squeeze(0)
        # item_A = torch.stack((img_f.squeeze(0), img_l.squeeze(0)), dim=0)

        return {'A': item_A, 'path': self.files_A[index % len(self.files_A)]}

    def __len__(self):
        return len(self.files_A)