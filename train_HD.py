#!/usr/bin/python3
from pathlib import Path
import argparse
import itertools

import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
from torch import nn
import os

from models.model import LocalEnhancer2Dto3D, LocalEnhancer3Dto2D
from models.model import Discriminator2D, Discriminator3D
from utils import ReplayBuffer
from utils import LambdaLR
from utils import weights_init_normal, load_network
from loader import ImageDataset

scaler = torch.cuda.amp.GradScaler()

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [2, 1, 0]))

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=6, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='../datasets/xray2ct/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=128, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--n_cpu', type=int, default=16, help='number of cpu threads to use during batch generation')
parser.add_argument('--fp16', type=bool, default=False, help='use mixed precision or not')
parser.add_argument('--generator_A2B', type=str, default='output/test_64_noide/netG_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='output/test_64_noide/netG_B2A.pth', help='B2A generator checkpoint file')
parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers used')
parser.add_argument('--niter_fix_global', type=int, default=20, help='number of epochs that we only train the outmost local enhancer')
opt = parser.parse_args()
print(opt)

###### Definition of variables ######s
# Networks
netG_A2B = LocalEnhancer2Dto3D(opt.input_nc, opt.output_nc, f_maps=16, num_levels=6, n_local_enhancers=opt.n_local_enhancers)
netG_B2A = LocalEnhancer3Dto2D(opt.output_nc, opt.input_nc, f_maps=16, num_levels=6, n_local_enhancers=opt.n_local_enhancers)
netD_A = Discriminator2D(opt.input_nc)
netD_B = Discriminator3D(opt.output_nc)

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

# load pretrained models
netG_A2B = load_network(netG_A2B, opt.generator_A2B)
netG_B2A = load_network(netG_B2A, opt.generator_B2A)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()

# Freeze global generators 
finetune_list = set()
params_dict = dict(netG_A2B.named_parameters())
params_GA2B = []
for key, value in params_dict.items():       
    if key.startswith('module.model' + str(opt.n_local_enhancers)):                    
        params_GA2B += [value]
        finetune_list.add(key)  
print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
print('The layers that are finetuned in G A2B are ', sorted(finetune_list))   

finetune_list = set()
params_dict = dict(netG_B2A.named_parameters())
params_GB2A = []
for key, value in params_dict.items():       
    if key.startswith('module.model' + str(opt.n_local_enhancers)):                    
        params_GB2A += [value]
        finetune_list.add(key)  
print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
print('The layers that are finetuned in G B2A are ', sorted(finetune_list))   

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(params_GA2B, params_GB2A),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size, opt.size)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [ transforms.Resize(int(opt.size * 1.18), Image.BICUBIC), 
                transforms.RandomCrop(opt.size), 
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True), 
                        batch_size=opt.batchSize, shuffle=True, drop_last=True, 
                        pin_memory=True, num_workers=opt.n_cpu)

# Loss plot
writer = SummaryWriter(comment=f'LR_{opt.lr}_BS_{opt.batchSize}')
###################################
train_iter = 0
###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # GAN loss
        fake_B = netG_A2B(real_A) # xray -> CT
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)
        
        fake_A = netG_B2A(real_B) # CT -> xray
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B) # xray -> CT -> xray
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0
        
        recovered_B = netG_A2B(fake_A) # CT -> xray -> CT
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*30.0

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

        writer.add_scalar('G_Loss/'+'train', loss_G, train_iter)
        writer.add_scalar('G_GANLoss/'+'train', loss_GAN_A2B + loss_GAN_B2A, train_iter)
        writer.add_scalar('G_cycleLoss/'+'train', loss_cycle_ABA + loss_cycle_BAB, train_iter)
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A) # D(xray)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

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
        loss_D_real = criterion_GAN(pred_real, target_real)
        
        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

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

        writer.add_scalar('D_Loss/'+'train', loss_D_A+loss_D_B, train_iter)
        writer.add_scalar('2D_DLoss/'+'train', loss_D_A, train_iter)
        writer.add_scalar('3D_DLoss/'+'train', loss_D_B, train_iter)
        train_iter += 1
        real_grid = torchvision.utils.make_grid((real_B[:,:,64]+1)/2)
        writer.add_image('Real CTs', real_grid)
        fake_grid = torchvision.utils.make_grid((fake_B[:,:,64]+1)/2)
        writer.add_image('Generated CTs', fake_grid)


    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    if opt.fp16:
        scaler.update()

    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
        print('------------ Now also finetuning global generator -----------')

    # Save models checkpoints
    path = './output/test_/'
    Path(path).mkdir(parents=True, exist_ok=True)
    torch.save(netG_A2B.state_dict(), path + 'netG_A2B.pth')
    torch.save(netG_B2A.state_dict(), path + 'netG_B2A.pth')
    torch.save(netD_A.state_dict(), path + 'netD_A.pth')
    torch.save(netD_B.state_dict(), path + 'netD_B.pth')

writer.close()
###################################