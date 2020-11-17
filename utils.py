import random
import time
import datetime
import sys
import math
import os
from collections import OrderedDict

from torch import autograd
from torch.autograd import Variable
import torch
import torch.nn as nn
from visdom import Visdom
import numpy as np
import SimpleITK as sitk

def save_numpy(tensor, name):
    array = torch.squeeze(tensor).cpu().float().numpy()
    np.save(name, array)

def save_dicom(tensor, name):
    array = torch.squeeze(tensor).cpu().float().numpy()
    array = np.clip(np.rint(array * 255.0), 0.0, 255.0).astype(np.uint8)
    array = np.moveaxis(array, 1, 0)
    array = array[::-1]
    dicom_scan = sitk.GetImageFromArray(array)
    sitk.WriteImage(dicom_scan, name)

def load_network(network, save_path=None):          
    if not os.path.isfile(save_path):
        print('%s not exists yet!' % save_path)
    else:
        try:
            network.load_state_dict(torch.load(save_path))
        except:   
            saved_dict = torch.load(save_path)    
            pretrained_dict = OrderedDict()
            for k, v in saved_dict.items():
                ks = k.split('.')
                ks[1] = 'model.'+ks[1]
                name = '.'.join(ks)
                pretrained_dict[name] = v

            model_dict = network.state_dict()
            
            try:
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}                    
                network.load_state_dict(pretrained_dict)
                print('Pretrained network G has excessive layers; Only loading layers that are used')
            except:
                print('Pretrained network G has fewer layers; The following are not initialized:')
                for k, v in pretrained_dict.items():                    
                    if v.size() == model_dict[k].size():
                        model_dict[k] = v

                not_initialized = set()                  
                for k, v in model_dict.items():
                    if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                        # not_initialized.add(k.split('.')[0])
                        not_initialized.add(k)
                
                print(sorted(not_initialized))
                network.load_state_dict(model_dict)   
    return network

def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

class Logger():
    def __init__(self, n_epochs, batches_epoch):
        self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}


    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].item()
            else:
                self.losses[loss_name] += losses[loss_name].item()

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch 
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title':image_name})
            else:
                self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name], opts={'title':image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), 
                                                                    opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1

        

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal_(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

def weights_init_normal(net):
	for m in net.modules():
		if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
			n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
			m.weight.data.normal_(0, math.sqrt(2. / n))
			if m.bias is not None:
				m.bias.data.zero_()
		elif isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
			n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
			m.weight.data.normal_(0, math.sqrt(2. / n))
			if m.bias is not None:
				m.bias.data.zero_()
		elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
			m.weight.data.fill_(1)
			m.bias.data.zero_()
		elif isinstance(m, nn.Linear):
			m.weight.data.normal_(0, 0.01)
			m.bias.data.zero_()

def calc_gradient_penalty(netD, real_data, fake_data, data_dim, batch_size, dim, gp_lambda):
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
    if data_dim == '2d':
        alpha = alpha.view(batch_size, 3, dim, dim)
        fake_data = fake_data.view(batch_size, 3, dim, dim)
    elif data_dim == '3d':
        alpha = alpha.view(batch_size, 1, dim, dim, dim)
        fake_data = fake_data.view(batch_size, 1, dim, dim, dim)
    alpha = alpha.cuda()
    
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty