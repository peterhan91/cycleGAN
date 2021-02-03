import glob
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        # full path list of xrays 
        self.files_A = sorted(glob.glob(os.path.join(root, '%s/xrays/frontal' % mode) + '/*.png'))[2:10000]
        self.files_C = sorted(glob.glob(os.path.join(root, '%s/xrays/lateral' % mode) + '/*.png'))[2:10000]
        # full folder list of cts
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/cts/data' % mode) + '/*'))

    def __getitem__(self, index):
        img_f = Image.open(self.files_A[index % len(self.files_A)])
        img_l = Image.open(self.files_C[index % len(self.files_C)])
        img_f = self.transform(img_f)
        img_l = self.transform(img_l)
        item_A = torch.stack((img_f.squeeze(0), img_l.squeeze(0)), dim=0)

        item_B = np.float32(np.load(self.files_B[random.randint(0, len(self.files_B) - 1)] + '/128.npy'))
        while not item_B.shape == (128, 128, 128):
            item_B = np.float32(np.load(self.files_B[random.randint(0, len(self.files_B) - 1)] + '/128.npy'))
        item_B = np.moveaxis(np.flipud(item_B), 1, 0)

        return {'A': item_A, 
                'B': torch.from_numpy(np.expand_dims(item_B.copy(), axis=0)).type(torch.FloatTensor),
                }

    def __len__(self):
        return min(len(self.files_A), len(self.files_B))


class TestDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.files_A = sorted(glob.glob(os.path.join(root, '%s/xrays/frontal' % mode) + '/*.png'))[:20]
        self.files_C = sorted(glob.glob(os.path.join(root, '%s/xrays/lateral' % mode) + '/*.png'))[:20]

    def __getitem__(self, index):
        img_f = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        img_l = self.transform(Image.open(self.files_C[index % len(self.files_C)]))
        item_A = torch.stack((img_f.squeeze(0), img_l.squeeze(0)), dim=0)

        return {'A': item_A, 'path': self.files_A[index % len(self.files_A)]}

    def __len__(self):
        return len(self.files_A)