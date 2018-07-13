from torch.utils.data import *
import torch.nn.functional as F
import torch
import torch.nn as nn

############################################################################
# Re-usable blocks
############################################################################


class ConvTrans(nn.Module):
    # One Block to be used as conv and transpose throughout the model
    def __init__(self, ic=4, oc=4, kernel_size=4, block_type='res', padding=1, stride=2, drop=.01, use_bn=True):
        super(ConvTrans, self).__init__()
        self.block_type = block_type

        operations = []
        operations += [nn.LeakyReLU(.2, inplace=True)]

        if self.block_type == 'up':
            operations += [nn.ConvTranspose2d(in_channels=ic, out_channels=oc, padding=padding,
                                              kernel_size=kernel_size, stride=stride, bias=False)]

        elif self.block_type == 'down':
            operations += [nn.Conv2d(in_channels=ic, out_channels=oc, padding=padding, kernel_size=kernel_size,
                                     stride=stride, bias=False)]

        if use_bn:
            operations += [nn.BatchNorm2d(oc)]

        operations += [nn.Dropout(drop)]

        self.block = nn.Sequential(*operations)

    def forward(self, x):
        return self.block(x)


############################################################################
# Generator and Discriminator
############################################################################


class Generator(nn.Module):
    # Generator with skip connections
    def __init__(self, layers=3, filts=1024, kernel_size=4, channels=3):
        super(Generator, self).__init__()

        up = []
        down = []
        filt = 32

        # our input and our output
        out = [ConvTrans(ic=2 * filt, oc=channels, kernel_size=kernel_size,
                         block_type='up', use_bn=False, drop=0)]
        inp = [nn.Conv2d(in_channels=channels, out_channels=filt, padding=1,
                         kernel_size=kernel_size, stride=2)]

        # conv and trans building from outside in, constraining max filter size using min()
        for a in range(layers):
            drop = .01
            if a > layers - 4:
                drop = .5

            down += [ConvTrans(ic=min([filt, filts]), oc=min([filt * 2, filts]),
                               kernel_size=kernel_size, block_type='down')]
            up += [ConvTrans(ic=min([filt * 4, filts * 2]), oc=min([filt, filts]), kernel_size=kernel_size,
                             block_type='up', drop=drop)]
            filt = int(filt * 2)

        # bottleneck where we reach 1x1
        core = [ConvTrans(ic=min([filt, filts]), oc=min([filt, filts]), kernel_size=kernel_size, block_type='down',
                          use_bn=False),
                ConvTrans(ic=min([filt, filts]), oc=min([filt, filts]), kernel_size=kernel_size, block_type='up')]

        up.reverse()
        down = inp + down
        up = up + out

        self.down = nn.ModuleList(down)
        self.core = nn.Sequential(*core)
        self.up = nn.ModuleList(up)

    def forward(self, x):
        skip_list = []
        # collect skips from conv operations
        for i in range(len(self.down)):
            x = self.down[i](x)
            skip_list.append(x)

        skip_list.reverse()
        x = self.core(x)
        # concatenate skips, and then feed to transpose operations
        for i in range(len(self.up)):
            x = self.up[i](torch.cat([x, skip_list[i]], 1))

        return F.tanh(x)


class Discriminator(nn.Module):
    def __init__(self, channels=3, filts=512, kernel_size=4, layers=5):
        super(Discriminator, self).__init__()

        operations = []
        out_operations = [nn.Conv2d(in_channels=filts, out_channels=1, padding=0, kernel_size=kernel_size, stride=1)]

        # Build up discriminator backwards based on final filter count
        for a in range(layers):
            if a == layers - 1:
                operations += [ConvTrans(ic=channels, oc=filts, kernel_size=kernel_size, block_type='down')]
            else:
                operations += [ConvTrans(ic=int(filts // 2), oc=filts, kernel_size=kernel_size, block_type='down')]
            filts = int(filts // 2)

        operations.reverse()

        operations += out_operations
        self.operations = nn.Sequential(*operations)

    def forward(self, x):
        return F.sigmoid(self.operations(x))



