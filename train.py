#!/usr/bin/python3
from pathlib import Path
import argparse
import itertools

from torch import nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import os

from models.model import Generator2Dto3D, Generator3Dto2D
from models.model import Discriminator2D, Discriminator3D
from utils import ReplayBuffer
from utils import LambdaLR
from utils import weights_init_normal
from loader import ImageDataset

scaler = torch.cuda.amp.GradScaler()

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [1, 2, 0]))

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=3, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='../../', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=70, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=128, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--n_cpu', type=int, default=16, help='number of cpu threads to use during batch generation')
parser.add_argument('--fp16', type=bool, default=False, help='use mixed precision or not')
opt = parser.parse_args()
print(opt)

###### Definition of variables ######s
# Networks
netG_A2B = Generator2Dto3D(opt.input_nc+2, opt.output_nc, f_maps=16, num_levels=4)
netG_B2A = Generator3Dto2D(opt.output_nc, opt.input_nc+2, f_maps=16, num_levels=4)
netD_A = Discriminator2D(opt.input_nc+2)
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

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_cycle_ = torch.nn.L1Loss()
criterion_idt = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), 
                                            netG_B2A.parameters()),
                                lr=opt.lr, betas=(0.9, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.9, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(),
                                lr=opt.lr, betas=(0.9, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, 
                    lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, 
                    lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, 
                    lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor
# input_A = Tensor(opt.batchSize, opt.input_nc+2, opt.size, opt.size)
# input_B = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size, opt.size)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

transforms_ = [ transforms.Resize(128, Image.BICUBIC), 
                # transforms.RandomCrop(opt.size), 
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)) ]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True), 
                        batch_size=opt.batchSize, shuffle=True, drop_last=True, 
                        pin_memory=True, num_workers=opt.n_cpu, collate_fn=collate_fn)

###################################
train_iter = 0
lambda_gan = 1
lambda_cycle = 10
lambda_ident = 0.5

writer = SummaryWriter(comment=f'LR_{opt.lr}_BS_{opt.batchSize}_lambg_{lambda_gan}_lambc_{lambda_cycle}')
###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input
        # real_A = Variable(input_A.copy_(batch['A']))
        # real_B = Variable(input_B.copy_(batch['B']))
        real_A = Variable(batch['A']).cuda()
        real_B = Variable(batch['B']).cuda()
        target_real = Variable(Tensor(real_A.shape[0], 1).fill_(1.0), requires_grad=False)
        target_fake = Variable(Tensor(real_B.shape[0], 1).fill_(0.0), requires_grad=False)

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # GAN loss
        fake_B = netG_A2B(real_A) # xray -> CT
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real) * lambda_gan
        
        fake_A = netG_B2A(real_B) # CT -> xray
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real) * lambda_gan

        # Cycle loss
        recovered_A = netG_B2A(fake_B) # xray -> CT -> xray
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*lambda_cycle

        recovered_B = netG_A2B(fake_A) # CT -> xray -> CT
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*lambda_cycle

        loss_later = criterion_cycle_(recovered_B.mean(-1), real_B.mean(-1))
        loss_axial = criterion_cycle_(recovered_B.mean(-2), real_B.mean(-2))
        loss_front = criterion_cycle_(recovered_B.mean(-3), real_B.mean(-3))
        loss_cycle_proj = (loss_axial + loss_front + loss_later) / 3 
        loss_cycle_BAB += loss_cycle_proj * lambda_cycle
    
        # identity loss
        proj_B = real_B.repeat(1, 3, 1, 1, 1).mean(-3)
        proj_B -= proj_B.min(-1, keepdim=True)[0]
        proj_B /= proj_B.max(-1, keepdim=True)[0]
        ident_B = netG_A2B((proj_B-0.5)/0.5)
        loss_ident_B2B = criterion_idt(ident_B, real_B) * lambda_ident
        
        # Total loss        
        if opt.fp16:
            with torch.cuda.amp.autocast():
                loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + loss_ident_B2B
            scaler.scale(loss_G).backward()
            scaler.step(optimizer_G)
        else:
            loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + loss_ident_B2B
            loss_G.backward()
            optimizer_G.step()

        print(f"G Loss: {loss_G} G GAN Loss: {loss_GAN_A2B + loss_GAN_B2A} G cycle Loss: {loss_cycle_ABA + loss_cycle_BAB}")
        writer.add_scalar('G_Loss/'+'train', loss_G, train_iter)
        writer.add_scalar('G_GANLoss/'+'train', loss_GAN_A2B + loss_GAN_B2A, train_iter)
        writer.add_scalar('G_cycleLoss/'+'train', loss_cycle_ABA + loss_cycle_BAB, train_iter)
        writer.add_scalar('G_identityLoss/'+'train', loss_ident_B2B, train_iter)
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A) # D(xray)
        # pred_real_ = netD_A(real_A[:,1,...].unsqueeze(1))
        loss_D_real = criterion_GAN(pred_real, target_real)
        # loss_D_real += criterion_GAN(pred_real_, target_real)
        # loss_D_real = loss_D_real / 2

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        # fake_A_ = fake_A_buffer_.push_and_pop(fake_A_)
        pred_fake = netD_A(fake_A.detach())
        # pred_fake_ = netD_A(fake_A_.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)
        # loss_D_fake += criterion_GAN(pred_fake_, target_fake)
        # loss_D_fake = loss_D_fake / 2

        # Total loss
        if opt.fp16:
            with torch.cuda.amp.autocast():
                loss_D_A = (loss_D_real + loss_D_fake)*0.05
            scaler.scale(loss_D_A).backward()
            scaler.step(optimizer_D_A)
        else:
            loss_D_A = (loss_D_real + loss_D_fake) * lambda_gan
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
            loss_D_B = (loss_D_real + loss_D_fake) * lambda_gan
            loss_D_B.backward()
            optimizer_D_B.step()
        ###################################

        print(f"D Loss: {loss_D_A + loss_D_B}")
        writer.add_scalar('D_Loss/'+'train', loss_D_A+loss_D_B, train_iter)
        writer.add_scalar('2D_DLoss/'+'train', loss_D_A, train_iter)
        writer.add_scalar('3D_DLoss/'+'train', loss_D_B, train_iter)
        train_iter += 1
        real_grid = torchvision.utils.make_grid(real_A)
        writer.add_image('Real X-rays', 0.5*(real_grid+1))
        fake_grid = torchvision.utils.make_grid(fake_A)
        writer.add_image('Generated X-rays', 0.5*(fake_grid+1))


    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    if opt.fp16:
        scaler.update()

    # Save models checkpoints
    path = './output/test_10_1/'
    Path(path).mkdir(parents=True, exist_ok=True)
    torch.save(netG_A2B.state_dict(), path + 'netG_A2B.pth')
    torch.save(netG_B2A.state_dict(), path + 'netG_B2A.pth')
    torch.save(netD_A.state_dict(), path + 'netD_A.pth')
    torch.save(netD_B.state_dict(), path + 'netD_B.pth')

writer.close()
###################################