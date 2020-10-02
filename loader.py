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
        self.files_A = sorted(glob.glob(os.path.join(root, '%s/xray' % mode) + '/*.png'))
        # full folder list of cts
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/ct' % mode) + '/*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))

        if self.unaligned:
            # item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
            item_B = np.float32(np.load(self.files_B[random.randint(0, len(self.files_B) - 1)] + '/128x128.npy'))
        else:
            # item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))
            item_B = np.float32(np.load(self.files_B[index % len(self.files_B)] + '/128x128.npy'))

        return {'A': item_A, 
                'B': torch.from_numpy(item_B).type(torch.FloatTensor)}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))