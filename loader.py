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
        self.files_A = sorted(glob.glob(os.path.join(root, '%s/stanford_xrays/frontal' % mode) + '/*.png'))[2:8000]
        # self.files_C = sorted(glob.glob(os.path.join(root, '%s/stanford_xrays/lateral' % mode) + '/*.png'))[2:7000]
        # full folder list of cts
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/ct_128_' % mode) + '/*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))
        # lateral = Image.open(self.files_C[index % len(self.files_C)]).convert('L')
        # item_C = lateral.transpose(Image.FLIP_LEFT_RIGHT).rotate(90)
        # item_C = self.transform(item_C)

        item_B = np.float32(np.load(self.files_B[random.randint(0, len(self.files_B) - 1)] + '/128x128.npy'))
        while not item_B.shape == (128, 128, 128):
            item_B = np.float32(np.load(self.files_B[random.randint(0, len(self.files_B) - 1)] + '/128x128.npy'))
        item_B = np.moveaxis(np.flipud(item_B), 1, 0)

        return {'A': item_A, 
                'B': torch.from_numpy(np.expand_dims(item_B.copy(), axis=0)).type(torch.FloatTensor),
                # 'C': item_C
                }

    def __len__(self):
        return min(len(self.files_A), len(self.files_B))


class TestDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.files_A = sorted(glob.glob(os.path.join(root, 'LIDC_xrays') + '/*'+'/1-1.png'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))

        return {'A': item_A, 'path': self.files_A[index % len(self.files_A)]}

    def __len__(self):
        return len(self.files_A)