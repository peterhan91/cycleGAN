#!/usr/bin/python3
from pathlib import Path
import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
from torch import nn
import os

from models.model import Generator2Dto3D, Generator3Dto2D
from models.model import WDiscriminator2D, WDiscriminator3D
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal, calc_gradient_penalty
from loader import ImageDataset

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [2, 1, 0]))

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=9, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='../datasets/xray2ct/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=128, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
print(opt)

###### Definition of variables ######
# Networks
netG_A2B = Generator2Dto3D(opt.input_nc, opt.output_nc, fmaps=64, num_levels=6)
netG_B2A = Generator3Dto2D(opt.output_nc, opt.input_nc, fmaps=64, num_levels=6)
netD_A = WDiscriminator2D(opt.input_nc)
netD_B = WDiscriminator3D(opt.output_nc)

netG_A2B = nn.DataParallel(netG_A2B)
netG_B2A = nn.DataParallel(netG_B2A)
netD_A = nn.DataParallel(netD_A)
netD_B = nn.DataParallel(netD_B)

netG_A2B.cuda()
netG_B2A.cuda()
netD_A.cuda()
netD_B.cuda()

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# Lossess
criterion_cycle = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G_A2B = torch.optim.Adam(netG_A2B.parameters(), lr=opt.lr, betas=(0.5, 0.9))
optimizer_G_B2A = torch.optim.Adam(netG_B2A.parameters(), lr=opt.lr, betas=(0.5, 0.9))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.9))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.9))

lr_scheduler_G_A2B = torch.optim.lr_scheduler.LambdaLR(optimizer_G_A2B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_G_B2A = torch.optim.lr_scheduler.LambdaLR(optimizer_G_B2A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size, opt.size)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [ transforms.Resize(int(opt.size*1.12), Image.BICUBIC), 
                transforms.RandomCrop(opt.size), 
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True), 
                        batch_size=opt.batchSize, shuffle=True, drop_last=True, 
                        pin_memory=True, num_workers=opt.n_cpu)

# Loss plot
logger = Logger(opt.n_epochs, len(dataloader))
###################################

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))
        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        one, mone = one.cuda(), mone.cuda()

        ###### Generators A2B and B2A ######
        optimizer_G_A2B.zero_grad()
        optimizer_G_B2A.zero_grad()

        for p in netD_A.parameters():
            p.requires_grad_(False)  # freeze D
        for p in netD_B.parameters():
            p.requires_grad_(False)  # freeze D

        # GAN (generator) loss
        fake_B = netG_A2B(real_A) # xray -> CT
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = pred_fake.mean()
        loss_GAN_A2B.backward(mone, retain_graph=True)
        loss_GAN_A2B = -1 * loss_GAN_A2B
        
        fake_A = netG_B2A(real_B) # CT -> xray
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = pred_fake.mean()
        loss_GAN_B2A.backward(mone, retain_graph=True)
        loss_GAN_B2A = -1 * loss_GAN_B2A

        # Cycle loss
        recovered_A = netG_B2A(fake_B) # xray -> CT -> xray
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0
        loss_cycle_ABA.backward()
        
        recovered_B = netG_A2B(fake_A) # CT -> xray -> CT
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0
        loss_cycle_BAB.backward()

        loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        # print(loss_G)
        
        optimizer_G_A2B.step()
        optimizer_G_B2A.step()
        ###################################

        ###### Discriminator A (2D X-ray) ###### 
        optimizer_D_A.zero_grad()
        for p in netD_A.parameters():
            p.requires_grad_(True)  # unfreeze D

        # Real loss
        pred_real = netD_A(real_A) # D(xray)
        loss_D_real = pred_real.mean()

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = pred_fake.mean()

        # Gradient penalty
        loss_GP = calc_gradient_penalty(netD_A, real_A, fake_A.detach(), '2d',
                                                opt.batchSize, opt.size, 10.0)
        loss_GP.backward(retain_graph=True)

        # Total loss
        loss_D_A = (loss_D_fake - loss_D_real)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B (3D CT) ######
        optimizer_D_B.zero_grad()
        for p in netD_B.parameters():
            p.requires_grad_(True)  # unfreeze D

        # Real loss
        pred_real = netD_B(real_B) #D(CT)
        loss_D_real = pred_real.mean()
        
        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = pred_fake.mean()

        # Gradient penalty
        loss_GP = calc_gradient_penalty(netD_B, real_B, fake_B.detach(), '3d',
                                                opt.batchSize, opt.size, 10.0)
        loss_GP.backward(retain_graph=True)        

        # Total loss
        loss_D_B = (loss_D_fake - loss_D_real)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################

        # Progress report (http://localhost:8097)
        logger.log({'loss_G': loss_G, 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                    'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)},
                    images={'real_A': real_A, 'real_B': torch.squeeze(real_B)[0], 'fake_A': fake_A, 'fake_B': torch.squeeze(fake_B)[0]}
                )

    # Update learning rates
    lr_scheduler_G_A2B.step()
    lr_scheduler_G_B2A.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save models checkpoints
    path = './output/wgan/'
    Path(path).mkdir(parents=True, exist_ok=True)
    torch.save(netG_A2B.state_dict(), path + 'netG_A2B.pth')
    torch.save(netG_B2A.state_dict(), path + 'netG_B2A.pth')
    torch.save(netD_A.state_dict(), path + 'netD_A.pth')
    torch.save(netD_B.state_dict(), path + 'netD_B.pth')
###################################