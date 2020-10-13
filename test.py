#!/usr/bin/python3

import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from torch import nn
from PIL import Image

from models.model import Generator2Dto3D, Generator3Dto2D
from loader import ImageDataset
from utils import save_numpy

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0, 1, 2]))

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='../datasets/xray2ct/', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--size', type=int, default=128, help='size of the data (squared assumed)')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='output/netG_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='output/netG_B2A.pth', help='B2A generator checkpoint file')
opt = parser.parse_args()
print(opt)

###### Definition of variables ######
# Networks
netG_A2B = Generator2Dto3D(opt.input_nc, opt.output_nc, fmaps=32, num_levels=6)
netG_A2B = nn.DataParallel(netG_A2B)
netG_A2B.cuda()

# Load state dicts
netG_A2B.load_state_dict(torch.load(opt.generator_A2B))

# Set model's test mode
netG_A2B.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor 
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)

# Dataset loader
transforms_ = [ transforms.Resize(int(opt.size), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test'), 
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
###################################

###### Testing######

# Create output dirs if they don't exist
if not os.path.exists('output/CT'):
    os.makedirs('output/CT')

for i, batch in enumerate(dataloader):
    # Set model input
    real_A = Variable(input_A.copy_(batch['A']))

    # Generate output
    fake_B = 0.5*(netG_A2B(real_A).data + 1.0)

    # Save image files
    save_numpy(fake_B, 'output/CT/%04d.npy' % (i+1))

    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

sys.stdout.write('\n')
###################################