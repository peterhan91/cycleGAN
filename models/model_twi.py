import torch.nn as nn
import torch.nn.functional as F
from models.buildingblocks import Encoder, Decoder, SingleConv, DoubleConv, Skipconnection

def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]

class Abstract3DUNet_plus(nn.Module):
    def __init__(self, in_channels, out_channels, final_sigmoid, basic_module, f_maps=64, layer_order='cr',
                 num_levels=4, is_segmentation=True, testing=False, en_kernel_type='2d', de_kernel_type='3d',
                 conv_kernel_size=3, pool_kernel_size=2, conv_padding=1, features_out=None, **kwargs):
        super(Abstract3DUNet_plus, self).__init__()

        self.testing = testing
        self.features_out = features_out
        self.encoder_type = en_kernel_type
        self.decoder_type = de_kernel_type

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
        decoders_ = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i]
            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder(in_feature_num, out_feature_num,
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              conv_kernel_type=de_kernel_type,
                              padding=conv_padding)
            decoder_ = Decoder(in_feature_num, out_feature_num,
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              conv_kernel_type=de_kernel_type,
                              padding=conv_padding)
            decoders.append(decoder)
            decoders_.append(decoder_)

        self.decoders = nn.ModuleList(decoders)
        self.decoders_ = nn.ModuleList(decoders_)

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
        encoders_features = []
        encoders_features_ = []
        if self.encoder_type == '2d': 
            x_frontal = x[:,0,...].unsqueeze(1)
            x_lateral = x[:,1,...].unsqueeze(1)
            for encoder in self.encoders:
                x_frontal = encoder(x_frontal)
                x_lateral = encoder(x_lateral)
                encoders_features.insert(0, x_frontal)
                encoders_features_.insert(0, x_lateral)         
        
        if self.encoder_type == '3d':
            for encoder in self.encoders:
                x = encoder(x)
                encoders_features.insert(0, x)
                encoders_features_.insert(0, x.rot90(1, [2, 4]))
            
            x_frontal = x
            x_lateral = x.rot90(1, [2, 4])

        encoders_features = encoders_features[1:]
        encoders_features_ = encoders_features_[1:]
        x_frontal = self.connect(x_frontal)
        x_lateral = self.connect(x_lateral)

        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x_frontal = decoder(encoder_features, x_frontal)
        
        for decoder_, encoder_features_ in zip(self.decoders_, encoders_features_):
            x_lateral = decoder_(encoder_features_, x_lateral)

        if self.features_out:
            return x_frontal, x_lateral
        
        elif self.decoder_type == '3d':
            x_frontal = self.final_conv(x_frontal)
            x_lateral = self.final_conv(x_lateral)
            if self.testing or self.final_activation is not None:
                x_frontal = self.final_activation(x_frontal)
                x_lateral = self.final_activation(x_lateral)
            return (x_lateral.rot90(1, [4, 2]) + x_frontal) * 0.5
        
        elif self.decoder_type == '2d':
            x_frontal = self.final_conv(x_frontal)
            x_lateral = self.final_conv(x_lateral)
            if self.testing or self.final_activation is not None:
                x_frontal = self.final_activation(x_frontal)
                x_lateral = self.final_activation(x_lateral)
            return x_frontal, x_lateral

class Generator2Dto3D(Abstract3DUNet_plus):
    def __init__(self, in_channels, out_channels, final_sigmoid=False, f_maps=16, layer_order='icr',
                 num_levels=4, is_segmentation=False, conv_padding=1, features_out=False, **kwargs):
        super(Generator2Dto3D, self).__init__(in_channels=in_channels, out_channels=out_channels, final_sigmoid=final_sigmoid,
                                     en_kernel_type='2d', de_kernel_type='3d',
                                     basic_module=DoubleConv, f_maps=f_maps, layer_order=layer_order,
                                     num_levels=num_levels, is_segmentation=is_segmentation,
                                     conv_padding=conv_padding, features_out=features_out, **kwargs)


class Generator3Dto2D(Abstract3DUNet_plus):
    def __init__(self, in_channels, out_channels, final_sigmoid=False, f_maps=16, layer_order='icr',
                 num_levels=4, is_segmentation=False, conv_padding=1, features_out=False, **kwargs):
        super(Generator3Dto2D, self).__init__(in_channels=in_channels, out_channels=out_channels, final_sigmoid=final_sigmoid,
                                     en_kernel_type='3d', de_kernel_type='2d',
                                     basic_module=DoubleConv, f_maps=f_maps, layer_order=layer_order,
                                     num_levels=num_levels, is_segmentation=is_segmentation,
                                     conv_padding=conv_padding, features_out=features_out, **kwargs)


class Discriminator2D(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator2D, self).__init__()
        self.base_nc = 64
        model = [   nn.Conv2d(input_nc, self.base_nc, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2) ]

        model += [  nn.Conv2d(self.base_nc, self.base_nc*2, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(self.base_nc*2), 
                    nn.LeakyReLU(0.2) ]

        model += [  nn.Conv2d(self.base_nc*2, self.base_nc*4, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(self.base_nc*4), 
                    nn.LeakyReLU(0.2) ]

        model += [  nn.Conv2d(self.base_nc*4, self.base_nc*8, 4, padding=1),
                    nn.InstanceNorm2d(self.base_nc*8), 
                    nn.LeakyReLU(0.2) ]

        model += [nn.Conv2d(self.base_nc*8, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
        return x


class Discriminator3D(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator3D, self).__init__()
        self.base_nc = 64
        model = [   nn.Conv3d(input_nc, self.base_nc, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2) ]

        model += [  nn.Conv3d(self.base_nc, self.base_nc*2, 4, stride=2, padding=1),
                    nn.InstanceNorm3d(self.base_nc*2), 
                    nn.LeakyReLU(0.2) ]

        model += [  nn.Conv3d(self.base_nc*2, self.base_nc*4, 4, stride=2, padding=1),
                    nn.InstanceNorm3d(self.base_nc*4), 
                    nn.LeakyReLU(0.2) ]

        model += [  nn.Conv3d(self.base_nc*4, self.base_nc*8, 4, padding=1),
                    nn.InstanceNorm3d(self.base_nc*8), 
                    nn.LeakyReLU(0.2) ]

        model += [nn.Conv3d(self.base_nc*8, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        x = F.avg_pool3d(x, x.size()[2:]).view(x.size()[0], -1)
        return x

