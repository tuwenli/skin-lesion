import torch.nn as nn
import torch.nn.functional as F
import torch
from numpy.random import normal
from math import sqrt
import argparse
import numpy as np
from model.layers import  Dense_layer,Transition_Down,Transition_Up

channel_dim = 3
ndf = 64

class GlobalConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(GlobalConvBlock, self).__init__()
        pad0 = (kernel_size[0] - 1) / 2
        pad1 = (kernel_size[1] - 1) / 2
        self.conv_l1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r
        return x

class ResidualBlock(nn.Module):
    def __init__(self, indim):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(indim, int(indim*0.9), kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm2d(int(indim*0.9))
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(int(indim*0.9), int(indim*0.9), kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(int(indim*0.9))
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(int(indim*0.9),int(indim*0.9), kernel_size=1, bias=False)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.downsample=nn.Sequential()
        self.downsample=nn.Sequential(nn.Conv2d(indim,int(int(indim*0.9)),1,stride=1),nn.BatchNorm2d(int(int(indim*0.9))))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu1(residual)
        residual = self.conv2(residual)
        residual = self.relu2(residual)
        residual = self.conv3(residual)
        residual = self.relu3(residual)
        x=self.downsample(x)
        out = x + residual
        return out

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu1(residual)
        residual = self.conv2(residual)
        residual = self.relu2(residual)
        residual = self.conv3(residual)
        residual = self.relu3(residual)
        out = x + residual
        return out

class DenseBlock(nn.Module):
    def __init__(self, input_depth, n_filters=16, filter_size=3, n_layers_per_block=5, dropout_p=0.2):
        super(DenseBlock, self).__init__()
        self.dense_list = nn.ModuleList([])

        self.num_layers = n_layers_per_block

        for i in range(n_layers_per_block):
            dense_layer = Dense_layer(
                input_depth, n_filters, filter_size, dropout_p)
            self.dense_list.append(dense_layer)

            input_depth += n_filters

    def forward(self, input):
        concat_list = []

        input = input
        for i in range(self.num_layers):
            output = self.dense_list[i](input)
            concat_list.append(output)

            input = torch.cat((output, input), dim=1)

        output = torch.cat(concat_list, dim=1)
        return output

class NetS(nn.Module):
    def __init__(self, n_layers_list, n_pool, input_depth=3, n_first_conv_filters=48, n_filters=16, filter_size=3, dropout_p=0.2, n_classes=1):
        super(NetS, self).__init__()

        self.n_pool = n_pool
        self.n_layers_list = n_layers_list
        self.depth_list = []
        self.Resi_up=[557,442,340,248,192]
        self.input_dense_up=[278,221,170,124,96]

        self.dense_blocks = nn.ModuleList([])
        self.residualblocks_down=nn.ModuleList([])
        self.residualblocks_up=nn.ModuleList([])
        self.TD = nn.ModuleList([])
        self.TU = nn.ModuleList([])

        assert(len(n_layers_list) == 2 * n_pool + 1)

        self.first_conv = nn.Conv2d(
            input_depth, n_first_conv_filters, filter_size, 1, (filter_size-1)//2)
        input_depth = n_first_conv_filters

        count = 0
        for i in range(n_pool):

            dense_block = DenseBlock(
                input_depth, n_filters, filter_size, n_layers_list[i], dropout_p)
            input_depth = input_depth + n_layers_list[i] * n_filters
            residualblock_down=ResidualBlock(input_depth)
            input_depth=residualblock_down.conv3.out_channels
            transition_down = Transition_Down(
                input_depth, input_depth, dropout_p)
            self.dense_blocks.append(dense_block)
            self.depth_list.append(input_depth)
            self.residualblocks_down.append(residualblock_down)
            self.TD.append(transition_down)
        self.dense_blocks.append(DenseBlock(
            input_depth, n_filters, filter_size, n_layers_list[self.n_pool], dropout_p))
        input_depth = input_depth + n_filters * \
            n_layers_list[self.n_pool]
        self.depth_list.append(input_depth)
        self.depth_list = self.depth_list[::-1]

        for idx, i in enumerate(range(self.n_pool+1, len(n_layers_list), 1)):
            n_filters_keep = n_filters * \
                n_layers_list[i-1]
            transtion_up = Transition_Up(n_filters_keep, n_filters_keep)
            residualblock_up=ResidualBlock(self.Resi_up[idx])
            dense_block = DenseBlock(
                self.input_dense_up[idx], n_filters, filter_size, n_layers_list[i], dropout_p)
            self.TU.append(transtion_up)
            self.residualblocks_up.append(residualblock_up)
            self.dense_blocks.append(dense_block)
        self.last_conv = nn.Conv2d(
            n_layers_list[-1]*n_filters, n_classes, 1, 1)

    def forward(self, input):
        cache_list = []
        first_conv = self.first_conv(input)
        _in = first_conv
        for i in range(self.n_pool):
            out = self.dense_blocks[i](_in)
            out = torch.cat((out, _in), dim=1)
            cache_list.append(out)
            out=self.residualblocks_down[i](out)
            out = self.TD[i](out)
            _in = out
        out = self.dense_blocks[self.n_pool](_in)
        _in = out
        cache_list = cache_list[::-1]
        for idx, i in enumerate(range(self.n_pool+1, len(self.dense_blocks), 1)):
            out = self.TU[idx](_in)
            out = self.residualblocks_up[idx](out)
            out = torch.cat((out, cache_list[idx]), dim=1)
            out = self.dense_blocks[i](out)
            _in = out
        out = self.last_conv(_in)
        out = F.softmax(out, dim=1)
        return out

class NetC(nn.Module):
    def __init__(self, ngpu):
        super(NetC, self).__init__()
        self.ngpu = ngpu
        self.convblock1 = nn.Sequential(
            nn.Conv2d(channel_dim, ndf, 7, 2, 3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.convblock1_1 = nn.Sequential(
            GlobalConvBlock(ndf, ndf * 2, (13, 13)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 2, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.convblock2_1 = nn.Sequential(
            GlobalConvBlock(ndf * 2, ndf * 2, (11, 11)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.convblock3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.convblock3_1 = nn.Sequential(
            GlobalConvBlock(ndf * 4, ndf * 4, (9, 9)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.convblock4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.convblock4_1 = nn.Sequential(
            GlobalConvBlock(ndf * 8, ndf * 8, (7, 7)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.convblock5 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.convblock5_1 = nn.Sequential(
            GlobalConvBlock(ndf * 8, ndf * 8, (5, 5)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu is 1:
            batchsize = input.size()[0]
            out1 = self.convblock1(input)
            out1=self.convblock1_1(out1)
            out2 = self.convblock2(out1)
            out2 = self.convblock2_1(out2)
            out3 = self.convblock3(out2)
            out3 = self.convblock3_1(out3)
            out4 = self.convblock4(out3)
            out4 = self.convblock4_1(out4)
            out5 = self.convblock5(out4)
            out5 = self.convblock5_1(out5)
            a=input.view(batchsize,-1)
            b=1*out1.view(batchsize,-1)
            c=2*out2.view(batchsize,-1)
            d=4*out5.view(batchsize,-1)
            output = torch.cat((input.view(batchsize,-1),1*out1.view(batchsize,-1),
                                2*out2.view(batchsize,-1),2*out3.view(batchsize,-1),
                                2*out4.view(batchsize,-1),4*out5.view(batchsize,-1)),1)
        else:
            print('For now we only support one GPU')
        return output

