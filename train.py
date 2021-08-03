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
import os

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from models.model import Generator2Dto3D, Generator3Dto2D
from models.model import Discriminator2D, Discriminator3D
from utils import ReplayBuffer
from utils import LambdaLR
from loader import ImageDataset

scaler = torch.cuda.amp.GradScaler()
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [1, 2, 0]))

def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=3, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='../../', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=128, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--n_cpu', type=int, default=32, help='number of cpu threads to use during batch generation')
parser.add_argument('--fp16', type=bool, default=False, help='use mixed precision or not')
parser.add_argument('--last_iter', type=int, default=0, help='last training iteration')
parser.add_argument('--checkpoint_path', type=str, default=None, help='checkpoint dir')

parser.add_argument('--logging', type=bool, default=True, help='logging with tensorboard')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--dist', type=bool, default=False)
parser.add_argument('--launcher', choices=['none', 'pytorch'], default='pytorch', help='job launcher')
opt = parser.parse_args()
print(opt)

#### distributed training settings ####
if opt.launcher == 'none':  # disabled distributed training
    opt.dist = False
    rank = -1
    print('Disabled distributed training.')
else:
    opt.dist = True
    init_dist()
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()


###### Definition of variables ######
torch.backends.cudnn.benchmark = True
device = torch.device('cuda')
# Networks
netG_A2B = Generator2Dto3D(opt.input_nc, opt.output_nc, f_maps=16, num_levels=6).to(device)
netG_B2A = Generator3Dto2D(opt.output_nc, opt.input_nc, f_maps=16, num_levels=6).to(device)
netD_A = Discriminator2D(opt.input_nc).to(device)
netD_B = Discriminator3D(opt.output_nc).to(device)


# Set networks to DistributedDataParallel
netG_A2B = DistributedDataParallel(netG_A2B, device_ids=[torch.cuda.current_device()])
netG_B2A = DistributedDataParallel(netG_B2A, device_ids=[torch.cuda.current_device()])
netD_A = DistributedDataParallel(netD_A, device_ids=[torch.cuda.current_device()])
netD_B = DistributedDataParallel(netD_B, device_ids=[torch.cuda.current_device()])


#### loading resume state if exists
if opt.checkpoint_path is not None:
    # distributed resuming: all load into default GPU
    device_id = torch.cuda.current_device()
    netG_A2B.load_state_dict(torch.load(os.path.join(opt.checkpoint_path, 'netG_A2B_%d.pth' %(opt.last_iter)),
                                    map_location=lambda storage, loc: storage.cuda(device_id)))
    netG_B2A.load_state_dict(torch.load(os.path.join(opt.checkpoint_path, 'netG_B2A_%d.pth' %(opt.last_iter)),
                                    map_location=lambda storage, loc: storage.cuda(device_id)))
    netD_A.load_state_dict(torch.load(os.path.join(opt.checkpoint_path, 'netD_A_%d.pth' %(opt.last_iter)),
                                    map_location=lambda storage, loc: storage.cuda(device_id)))
    netD_B.load_state_dict(torch.load(os.path.join(opt.checkpoint_path, 'netD_B_%d.pth' %(opt.last_iter)),
                                    map_location=lambda storage, loc: storage.cuda(device_id)))
    if dist.get_rank() == 0:
        print('resume training from iteration %d' %(opt.last_iter))


# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_cycle_ = torch.nn.L1Loss()
criterion_idt = torch.nn.L1Loss()


# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), 
                                            netG_B2A.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, 
                    lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, 
                    lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, 
                    lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)


# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor
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

my_trainset = ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True)
train_sampler = torch.utils.data.distributed.DistributedSampler(my_trainset)
dataloader = DataLoader(my_trainset, batch_size=opt.batchSize, pin_memory=True, drop_last=True,
                        num_workers=opt.n_cpu, collate_fn=collate_fn, sampler=train_sampler)


###################################
train_iter = opt.last_iter
lambda_gan2D = 0.5
lambda_gan3D = 0.5
lambda_cycle2D = 10
lambda_cycle3D = 20
lambda_ident = 0.5
if rank <= 0 and opt.logging:
    writer = SummaryWriter(comment=f'LR_{opt.lr}_BS_{opt.batchSize}_lambg_{lambda_gan3D}_lambc_{lambda_cycle3D}')
###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input
        # real_A = Variable(input_A.copy_(batch['A']))
        # real_B = Variable(input_B.copy_(batch['B']))
        real_A = batch['A'].to(device)
        real_B = batch['B'].to(device)
        target_real = Variable(Tensor(real_A.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
        target_fake = Variable(Tensor(real_B.shape[0], 1).fill_(0.0), requires_grad=False).to(device)

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # GAN loss
        fake_B = netG_A2B(real_A) # xray -> CT
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real) * lambda_gan2D
        fake_A = netG_B2A(real_B) # CT -> xray
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real) * lambda_gan3D

        # Cycle loss
        recovered_A = netG_B2A(fake_B) # xray -> CT -> xray
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*lambda_cycle2D
        recovered_B = netG_A2B(fake_A) # CT -> xray -> CT
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*lambda_cycle3D

        # loss_later = criterion_cycle_(recovered_B.mean(-1), real_B.mean(-1))
        # loss_axial = criterion_cycle_(recovered_B.mean(-2), real_B.mean(-2))
        # loss_front = criterion_cycle_(recovered_B.mean(-3), real_B.mean(-3))
        # loss_cycle_proj = (loss_axial + loss_front + loss_later) / 3 
        # loss_cycle_BAB += loss_cycle_proj * lambda_cycle2D
    
        # identity loss
        # proj_B = real_B.repeat(1, 3, 1, 1, 1).mean(-3)
        # proj_B -= proj_B.min(-1, keepdim=True)[0]
        # proj_B /= proj_B.max(-1, keepdim=True)[0]
        # ident_B = netG_A2B((proj_B-0.5)/0.5)
        # loss_ident_B2B = criterion_idt(ident_B, real_B) * lambda_ident
        
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
            loss_D_A = (loss_D_real + loss_D_fake) * lambda_gan2D
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
            loss_D_B = (loss_D_real + loss_D_fake) * lambda_gan3D
            loss_D_B.backward()
            optimizer_D_B.step()
        ###################################
        if dist.get_rank() == 0:
            print(f"G Loss: {loss_G} G GAN Loss: {loss_GAN_A2B + loss_GAN_B2A} G cycle Loss: {loss_cycle_ABA + loss_cycle_BAB}")
            print(f"D Loss: {loss_D_A + loss_D_B}")
            if opt.logging:
                writer.add_scalar('G_Loss/'+'train', loss_G, train_iter)
                writer.add_scalar('G_GANLoss/'+'train', loss_GAN_A2B + loss_GAN_B2A, train_iter)
                writer.add_scalar('G_cycleLoss/'+'train', loss_cycle_ABA + loss_cycle_BAB, train_iter)
                # writer.add_scalar('G_identityLoss/'+'train', loss_ident_B2B, train_iter)
                writer.add_scalar('D_Loss/'+'train', loss_D_A+loss_D_B, train_iter)
                writer.add_scalar('2D_DLoss/'+'train', loss_D_A, train_iter)
                writer.add_scalar('3D_DLoss/'+'train', loss_D_B, train_iter)
                if train_iter % 50 == 0:
                    with torch.no_grad():
                        real_grid = torchvision.utils.make_grid((real_B[:,:,64]+1)/2)
                        writer.add_image('Real CTs', real_grid, global_step=train_iter)
                        fake_grid = torchvision.utils.make_grid((fake_B[:,:,64]+1)/2)
                        writer.add_image('Generated CTs', fake_grid, global_step=train_iter)
            
            # Save models checkpoints
            if train_iter % 2000 == 0:
                path = './output/train/'
                Path(path).mkdir(parents=True, exist_ok=True)
                torch.save(netG_A2B.state_dict(), path + 'netG_A2B_%d.pth' %(train_iter))
                torch.save(netG_B2A.state_dict(), path + 'netG_B2A_%d.pth' %(train_iter))
                torch.save(netD_A.state_dict(), path + 'netD_A_%d.pth' %(train_iter))
                torch.save(netD_B.state_dict(), path + 'netD_B_%d.pth' %(train_iter))
            train_iter += 1


    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    if opt.fp16:
        scaler.update()


writer.close()
###################################