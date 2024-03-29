import torch
import torch.nn as nn
import torch.nn.functional as F
from models.buildingblocks import Encoder, Decoder, SingleConv, DoubleConv, Skipconnection

def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]


class Abstract3DUNet(nn.Module):

    def __init__(self, in_channels, out_channels, final_sigmoid, basic_module, f_maps=64, layer_order='cir',
                 num_levels=4, is_segmentation=True, testing=False, en_kernel_type='2d', de_kernel_type='3d',
                 conv_kernel_size=3, pool_kernel_size=2, conv_padding=1, features_out=None, **kwargs):
        super(Abstract3DUNet, self).__init__()

        self.testing = testing
        self.features_out = features_out

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        # create encoder path consisting of Encoder modules. Depth of the encoder is equal to `len(f_maps)`
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num,
                                  apply_pooling=False,  # skip pooling in the firs encoder
                                  basic_module=basic_module,
                                  conv_layer_order=layer_order,
                                  conv_kernel_size=conv_kernel_size,
                                  conv_kernel_type = en_kernel_type,
                                  padding=conv_padding)
            else:
                # TODO: adapt for anisotropy in the data, i.e. use proper pooling kernel to make the data isotropic after 1-2 pooling operations
                encoder = Encoder(f_maps[i - 1], out_feature_num,
                                  basic_module=basic_module,
                                  conv_layer_order=layer_order,
                                  conv_kernel_size=conv_kernel_size,
                                  conv_kernel_type=en_kernel_type,
                                  pool_kernel_size=pool_kernel_size,
                                  padding=conv_padding)

            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # the connection between encoder and decoder
        self.connect = Skipconnection(f_maps[-1], f_maps[-1], kernel_type=de_kernel_type)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            # if basic_module == DoubleConv:
            #     in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            # else:
            in_feature_num = reversed_f_maps[i]
            out_feature_num = reversed_f_maps[i + 1]
            # TODO: if non-standard pooling was used, make sure to use correct striding for transpose conv
            # currently strides with a constant stride: (2, 2, 2)
            decoder = Decoder(in_feature_num, out_feature_num,
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              conv_kernel_type=de_kernel_type,
                              padding=conv_padding)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        # self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        self.final_conv = SingleConv(f_maps[0], out_channels, kernel_size=1, 
                                    kernel_type=de_kernel_type, order='c', padding=0)

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = nn.Tanh()
    
    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # 2D to 3D or 3D to 2D connection
        x = self.connect(x)

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)
        
        if self.features_out:
            return x
        else:
            x = self.final_conv(x)

            # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs
            # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
            if self.testing or self.final_activation is not None:
                x = self.final_activation(x)

            return x


class Generator2Dto3D(Abstract3DUNet):
    def __init__(self, in_channels, out_channels, final_sigmoid=False, f_maps=16, layer_order='cir',
                 num_levels=4, is_segmentation=False, conv_padding=1, features_out=False, **kwargs):
        super(Generator2Dto3D, self).__init__(in_channels=in_channels, out_channels=out_channels, final_sigmoid=final_sigmoid,
                                     en_kernel_type='2d', de_kernel_type='3d',
                                     basic_module=DoubleConv, f_maps=f_maps, layer_order=layer_order,
                                     num_levels=num_levels, is_segmentation=is_segmentation,
                                     conv_padding=conv_padding, features_out=features_out, **kwargs)


class Generator3Dto2D(Abstract3DUNet):
    def __init__(self, in_channels, out_channels, final_sigmoid=False, f_maps=16, layer_order='cir',
                 num_levels=4, is_segmentation=False, conv_padding=1, features_out=False, **kwargs):
        super(Generator3Dto2D, self).__init__(in_channels=in_channels, out_channels=out_channels, final_sigmoid=final_sigmoid,
                                     en_kernel_type='3d', de_kernel_type='2d',
                                     basic_module=DoubleConv, f_maps=f_maps, layer_order=layer_order,
                                     num_levels=num_levels, is_segmentation=is_segmentation,
                                     conv_padding=conv_padding, features_out=features_out, **kwargs)


class LocalEnhancer2Dto3D(nn.Module):
    def __init__(self, input_nc, output_nc, f_maps, num_levels, n_local_enhancers=1, n_blocks_local=3):
            super(LocalEnhancer2Dto3D, self).__init__() 
            self.n_local_enhancers = n_local_enhancers
            self.model = Generator2Dto3D(input_nc, output_nc, f_maps=f_maps*2, num_levels=num_levels-1, features_out=True)
            self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)
        
            for n in range(1, n_local_enhancers+1):
                ### downsample            
                model_downsample = [nn.Conv2d(input_nc, f_maps//2, kernel_size=3, padding=1, bias=False), 
                                    nn.InstanceNorm2d(f_maps//2),
                                    nn.ReLU(True),
                                    nn.Conv2d(f_maps//2, f_maps, kernel_size=3, padding=1, bias=False), 
                                    nn.InstanceNorm2d(f_maps), 
                                    nn.ReLU(True)]
                ### residual blocks
                model_upsample = []
                # for i in range(n_blocks_local):
                #     model_upsample += [ExtResNetBlock(f_maps*2, f_maps*2, kernel_type='3d')]

                ### upsample
                model_upsample += [ nn.InstanceNorm3d(f_maps*2+f_maps), 
                                    nn.Conv3d(f_maps*2+f_maps, f_maps, kernel_size=3, padding=1, bias=False),
                                    nn.ReLU(True),
                                    nn.InstanceNorm3d(f_maps), 
                                    nn.Conv3d(f_maps, f_maps, kernel_size=3, padding=1, bias=False),
                                    nn.ReLU(True)
                                    ]      

                ### final convolution
                if n == n_local_enhancers:                
                    model_upsample += [ nn.Conv3d(f_maps, output_nc, kernel_size=1, padding=0), 
                                        nn.Tanh()]                       
                
                upsample = [nn.ConvTranspose3d(f_maps*2, f_maps*2, kernel_size=3, stride=2, padding=1, output_padding=1)]
                connect = Skipconnection(f_maps, f_maps, kernel_type='3d')
                
                setattr(self, 'model'+str(n)+'_01', nn.Sequential(*upsample))
                setattr(self, 'model'+str(n)+'_02', connect)
                setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
                setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))   

    def forward(self, input): 
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level with shape [?, 32, 64, 64, 64]
        output_prev = self.model(input_downsampled[-1])   
        
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')  
            upsample =  getattr(self, 'model'+str(n_local_enhancers)+'_01')
            connect =  getattr(self, 'model'+str(n_local_enhancers)+'_02')            
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers] 
            features = model_downsample(input_i)   
            output_prev = model_upsample(torch.cat((upsample(output_prev), connect(features)), dim=1))
        
        return output_prev            


class LocalEnhancer3Dto2D(nn.Module):
    def __init__(self, input_nc, output_nc, f_maps, num_levels, n_local_enhancers=1, n_blocks_local=3):
            super(LocalEnhancer3Dto2D, self).__init__() 
            self.n_local_enhancers = n_local_enhancers
            self.model = Generator3Dto2D(input_nc, output_nc, f_maps=f_maps*2, num_levels=num_levels-1, features_out=True)
            self.downsample = nn.AvgPool3d(1, stride=2, count_include_pad=False)
            
            for n in range(1, n_local_enhancers+1):
                ### downsample            
                model_downsample = [nn.Conv3d(input_nc, f_maps//2, kernel_size=3, padding=1, bias=False), 
                                    nn.InstanceNorm3d(f_maps//2),
                                    nn.ReLU(True),
                                    nn.Conv3d(f_maps//2, f_maps, kernel_size=3, padding=1, bias=False), 
                                    nn.InstanceNorm3d(f_maps), 
                                    nn.ReLU(True)]
                ### residual blocks
                model_upsample = []
                # for i in range(n_blocks_local):
                #     model_upsample += [ExtResNetBlock(f_maps*2, f_maps*2, kernel_type='2d')]

                ### upsample
                model_upsample += [ nn.InstanceNorm2d(f_maps*2+f_maps),
                                    nn.Conv2d(f_maps*2+f_maps, f_maps, kernel_size=3, padding=1, bias=False),
                                    nn.ReLU(True),
                                    nn.InstanceNorm2d(f_maps), 
                                    nn.Conv2d(f_maps, f_maps, kernel_size=3, padding=1, bias=False),
                                    nn.ReLU(True)
                                    ]      

                ### final convolution
                if n == n_local_enhancers:                
                    model_upsample += [ nn.Conv2d(f_maps, output_nc, kernel_size=1, padding=0), 
                                        nn.Tanh()]                       
                
                upsample = [nn.ConvTranspose2d(f_maps*2, f_maps*2, kernel_size=3, stride=2, padding=1, output_padding=1)]
                connect = Skipconnection(f_maps, f_maps, kernel_type='2d')
                setattr(self, 'model'+str(n)+'_01', nn.Sequential(*upsample))
                setattr(self, 'model'+str(n)+'_02', connect)   
                setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
                setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))   

    def forward(self, input): 
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])        
        
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')   
            upsample =  getattr(self, 'model'+str(n_local_enhancers)+'_01')
            connect =  getattr(self, 'model'+str(n_local_enhancers)+'_02')          
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]   
            features = model_downsample(input_i)   
            output_prev = model_upsample(torch.cat((upsample(output_prev), connect(features)), dim=1))
        
        return output_prev 


class Discriminator2D(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator2D, self).__init__()
        self.base_nc = 32
        # A bunch of convolutions one after another
        # input: N x channels_img x 128 x 128
        model = []
        model += [  nn.Conv2d(input_nc, self.base_nc, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2),
                    nn.Dropout2d(0.25) ] # 64x64
        model += [  nn.Conv2d(self.base_nc, self.base_nc*2, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(self.base_nc*2), 
                    nn.LeakyReLU(0.2),
                    nn.Dropout2d(0.25) ] # 32x32
        model += [  nn.Conv2d(self.base_nc*2, self.base_nc*4, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(self.base_nc*4), 
                    nn.LeakyReLU(0.2),
                    nn.Dropout2d(0.25) ] # 16x16
        self.model_last = nn.Sequential(nn.Conv2d(self.base_nc*4+1, self.base_nc*8, 4, stride=2, padding=1),
                                        nn.InstanceNorm2d(self.base_nc*8), 
                                        nn.LeakyReLU(0.2),
                                        nn.Dropout2d(0.25))# 8x8
        # model += [  nn.Conv2d(self.base_nc*8, self.base_nc*16, 4, stride=2, padding=1),
        #             nn.InstanceNorm2d(self.base_nc*16), 
        #             nn.LeakyReLU(0.2, inplace=True),
        #             nn.Dropout2d(0.25) ] # 4x4
        # FCN classification layer
        # model += [nn.Conv2d(self.base_nc*16, 1, 4, stride=2, padding=0)]
        self.classifier = nn.Linear(256, 1)
        self.model = nn.Sequential(*model)


    def minibatch_std(self, x):
        batch_statistics = (torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3]))
        # we take the std for each example (across all channels, and pixels) then we repeat it
        # for a single channel and concatenate it with the image. In this way the discriminator
        # will get information about the variation in the batch/image
        return torch.cat([x, batch_statistics], dim=1)


    def forward(self, x):
        x = self.model(x)
        x = self.minibatch_std(x)
        x = self.model_last(x)
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
        x = self.classifier(x)
        # Average pooling and flatten
        # x = F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
        return x


class Discriminator3D(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator3D, self).__init__()
        self.base_nc = 32
        # A bunch of convolutions one after another
        model = [   nn.Conv3d(input_nc, self.base_nc, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2),
                    nn.Dropout3d(0.25) ]
        model += [  nn.Conv3d(self.base_nc, self.base_nc*2, 4, stride=2, padding=1),
                    nn.InstanceNorm3d(self.base_nc*2), 
                    nn.LeakyReLU(0.2),
                    nn.Dropout3d(0.25) ]
        model += [  nn.Conv3d(self.base_nc*2, self.base_nc*4, 4, stride=2, padding=1),
                    nn.InstanceNorm3d(self.base_nc*4), 
                    nn.LeakyReLU(0.2),
                    nn.Dropout3d(0.25) ]

        self.model_last = nn.Sequential(nn.Conv3d(self.base_nc*4+1, self.base_nc*8, 4, stride=2, padding=1),
                                        nn.InstanceNorm3d(self.base_nc*8), 
                                        nn.LeakyReLU(0.2),
                                        nn.Dropout3d(0.25))
        # model += [  nn.Conv3d(self.base_nc*8, self.base_nc*16, 4, stride=2, padding=1),
        #             nn.InstanceNorm3d(self.base_nc*16), 
        #             nn.LeakyReLU(0.2, inplace=True),
        #             nn.Dropout3d(0.25) ]
        # FCN classification layer
        # model += [nn.Conv3d(self.base_nc*16, 1, 4, stride=2, padding=0)]
        self.classifier = nn.Linear(256, 1)
        self.model = nn.Sequential(*model)


    def minibatch_std(self, x):
        batch_statistics = (torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4]))
        # we take the std for each example (across all channels, and pixels) then we repeat it
        # for a single channel and concatenate it with the image. In this way the discriminator
        # will get information about the variation in the batch/image
        return torch.cat([x, batch_statistics], dim=1)


    def forward(self, x):
        x = self.model(x)
        x = self.minibatch_std(x)
        x = self.model_last(x)
        x = F.avg_pool3d(x, x.size()[2:]).view(x.size()[0], -1)
        x = self.classifier(x)
        # Average pooling and flatten
        # x = F.avg_pool3d(x, x.size()[2:]).view(x.size()[0], -1)
        return x

