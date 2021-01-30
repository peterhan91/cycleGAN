#!/usr/bin/python3
from pathlib import Path
import argparse
import itertools
import logging

from torch import nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import torch
import os

from models.model_twi import Generator2Dto3D, Generator3Dto2D
from models.model_twi import Discriminator2D, Discriminator3D
from utils import ReplayBuffer
from utils import LambdaLR
from utils import weights_init_normal
from loader import ImageDataset

scaler = torch.cuda.amp.GradScaler()

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [1, 2]))

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=2, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='../', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=128, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
parser.add_argument('--fp16', type=bool, default=False, help='use mixed precision or not')
opt = parser.parse_args()
print(opt)

###### Definition of variables ######s
# Networks
netG_A2B = Generator2Dto3D(opt.input_nc, opt.output_nc, f_maps=16, num_levels=6)
netG_B2A = Generator3Dto2D(opt.output_nc, opt.input_nc, f_maps=16, num_levels=6)
netD_A = Discriminator2D(opt.input_nc)
netD_B = Discriminator3D(opt.output_nc)
netD_B_2x = Discriminator3D(opt.output_nc)
netD_B_4x = Discriminator3D(opt.output_nc)

netG_A2B = nn.DataParallel(netG_A2B)
netG_B2A = nn.DataParallel(netG_B2A)
netD_A = nn.DataParallel(netD_A)
netD_B = nn.DataParallel(netD_B)
netD_B_2x = nn.DataParallel(netD_B_2x)
netD_B_4x = nn.DataParallel(netD_B_4x)

netG_A2B.cuda()
netG_B2A.cuda()
netD_A.cuda()
netD_B.cuda()
netD_B_2x.cuda()
netD_B_4x.cuda()

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)
netD_B_2x.apply(weights_init_normal)
netD_B_4x.apply(weights_init_normal)

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), 
                                            netG_B2A.parameters()),
                                lr=opt.lr, betas=(0.9, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.9, 0.999))
optimizer_D_B = torch.optim.Adam(itertools.chain(netD_B.parameters(), 
                                                netD_B_2x.parameters(), 
                                                netD_B_4x.parameters()),
                                    lr=opt.lr, betas=(0.9, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, 
                    lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, 
                    lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, 
                    lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor
input_A = Tensor(opt.batchSize, 2, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size, opt.size)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_A_buffer_ = ReplayBuffer() 
fake_B_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [ transforms.Resize(int(opt.size), Image.BICUBIC), 
                # transforms.RandomCrop(opt.size), 
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)) ]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True), 
                        batch_size=opt.batchSize, shuffle=True, drop_last=True, 
                        pin_memory=True, num_workers=opt.n_cpu)

writer = SummaryWriter(comment=f'LR_{opt.lr}_BS_{opt.batchSize}')
logging.info(f'''Starting training:
    Epochs:          {opt.n_epochs}
    Batch size:      {opt.batchSize}
    Learning rate:   {opt.lr}
''')

###################################
train_iter = 0
###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))
        # real_C = Variable(input_C.copy_(batch['C']))

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # GAN loss
        fake_B = netG_A2B(real_A) # xray -> CT
        pred_fake = netD_B(fake_B)
        pred_fake_2x = netD_B_2x(F.interpolate(fake_B, scale_factor=1/2, 
                                            mode='trilinear', align_corners=False, 
                                            recompute_scale_factor=True))
        pred_fake_4x= netD_B_4x(F.interpolate(fake_B, scale_factor=1/4, 
                                            mode='trilinear', align_corners=False, 
                                            recompute_scale_factor=True))
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)
        loss_GAN_A2B += criterion_GAN(pred_fake_2x, target_real)
        loss_GAN_A2B += criterion_GAN(pred_fake_4x, target_real)
        loss_GAN_A2B = loss_GAN_A2B / 3
        
        fake_A, fake_A_ = netG_B2A(real_B) # CT -> xray
        pred_fake = netD_A(fake_A)
        pred_fake_ = netD_A(fake_A_)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)
        loss_GAN_B2A += criterion_GAN(pred_fake_, target_real)
        loss_GAN_B2A = loss_GAN_B2A / 2

        # Cycle loss
        recovered_A, recovered_A_ = netG_B2A(fake_B) # xray -> CT -> xray
        loss_cycle_ABA = criterion_cycle(torch.cat((recovered_A, recovered_A_), dim=1), real_A)*10.0
        
        recovered_B = netG_A2B(torch.cat((fake_A, fake_A_), dim=1)) # CT -> xray -> CT
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*20.0
        
        # Total loss        
        if opt.fp16:
            with torch.cuda.amp.autocast():
                loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            scaler.scale(loss_G).backward()
            scaler.step(optimizer_G)
        else:
            loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB 
            loss_G.backward()
            optimizer_G.step()

        logging.info(f"G Loss: {loss_G} G GAN Loss: {loss_GAN_A2B + loss_GAN_B2A} G cycle Loss: {loss_cycle_ABA + loss_cycle_BAB}")
        writer.add_scalar('G_Loss/'+'train', loss_G, train_iter)
        writer.add_scalar('G_GANLoss/'+'train', loss_GAN_A2B + loss_GAN_B2A, train_iter)
        writer.add_scalar('G_cycleLoss/'+'train', loss_cycle_ABA + loss_cycle_BAB, train_iter)
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A[:,0,...].unsqueeze(1)) # D(xray)
        pred_real_ = netD_A(real_A[:,1,...].unsqueeze(1))
        loss_D_real = criterion_GAN(pred_real, target_real)
        loss_D_real += criterion_GAN(pred_real_, target_real)
        loss_D_real = loss_D_real / 2

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        fake_A_ = fake_A_buffer_.push_and_pop(fake_A_)
        pred_fake = netD_A(fake_A.detach())
        pred_fake_ = netD_A(fake_A_.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)
        loss_D_fake += criterion_GAN(pred_fake_, target_fake)
        loss_D_fake = loss_D_fake / 2

        # Total loss
        if opt.fp16:
            with torch.cuda.amp.autocast():
                loss_D_A = (loss_D_real + loss_D_fake)*0.05
            scaler.scale(loss_D_A).backward()
            scaler.step(optimizer_D_A)
        else:
            loss_D_A = (loss_D_real + loss_D_fake)*0.05
            loss_D_A.backward()
            optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B) #D(CT)
        pred_real_2x = netD_B_2x(F.interpolate(real_B, scale_factor=1/2, 
                                            mode='trilinear', align_corners=False, 
                                            recompute_scale_factor=True))
        pred_real_4x= netD_B_4x(F.interpolate(real_B, scale_factor=1/4, 
                                            mode='trilinear', align_corners=False, 
                                            recompute_scale_factor=True))
        loss_D_real = criterion_GAN(pred_real, target_real)
        loss_D_real += criterion_GAN(pred_real_2x, target_real)
        loss_D_real += criterion_GAN(pred_real_4x, target_real)
        loss_D_real = loss_D_real / 3
        
        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        pred_fake_2x = netD_B_2x(F.interpolate(fake_B.detach(), scale_factor=1/2, 
                                            mode='trilinear', align_corners=False, 
                                            recompute_scale_factor=True))
        pred_fake_4x= netD_B_4x(F.interpolate(fake_B.detach(), scale_factor=1/4, 
                                            mode='trilinear', align_corners=False, 
                                            recompute_scale_factor=True))
        loss_D_fake = criterion_GAN(pred_fake, target_fake)
        loss_D_fake += criterion_GAN(pred_fake_2x, target_fake)
        loss_D_fake += criterion_GAN(pred_fake_4x, target_fake)
        loss_D_fake = loss_D_fake / 3

        # Total loss
        if opt.fp16:
            with torch.cuda.amp.autocast():
                loss_D_B = (loss_D_real + loss_D_fake)*0.95
            scaler.scale(loss_D_B).backward()
            scaler.step(optimizer_D_B)
        else:
            loss_D_B = (loss_D_real + loss_D_fake)*0.95
            loss_D_B.backward()
            optimizer_D_B.step()
        ###################################

        logging.info(f"D Loss: {loss_D_A + loss_D_B}")
        writer.add_scalar('D_Loss/'+'train', loss_D_A+loss_D_B, train_iter)
        writer.add_scalar('2D_DLoss/'+'train', loss_D_A, train_iter)
        writer.add_scalar('3D_DLoss/'+'train', loss_D_B, train_iter)
        train_iter += 1
        real_grid = torchvision.utils.make_grid(real_A)
        writer.add_image('Real X-rays', real_grid)
        fake_grid = torchvision.utils.make_grid(fake_A)
        writer.add_image('Generated X-rays', fake_grid)


    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    if opt.fp16:
        scaler.update()

    # Save models checkpoints
    path = './output/test_/'
    Path(path).mkdir(parents=True, exist_ok=True)
    torch.save(netG_A2B.state_dict(), path + 'netG_A2B.pth')
    torch.save(netG_B2A.state_dict(), path + 'netG_B2A.pth')
    torch.save(netD_A.state_dict(), path + 'netD_A.pth')
    torch.save(netD_B.state_dict(), path + 'netD_B.pth')

writer.close()
###################################