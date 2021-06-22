"""
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] Cihang Xie, Yuxin Wu, Laurens van der Maaten, Alan Yuille, Kaiming He
    Feature Denoising for Improving Adversarial Robustness. arXiv:1812.03411
Explanation:
[1] If 'whether_denoising' is True, a ResNet with two denoising blocks will be created.
    In contrast 'whether_denoising' is False, a normal ResNet will be created.
[2] 'filter_type' decides which denoising operation the denoising block will apply.
    Now it includes 'Median_Filter' 'Mean_Filter' and 'Gaussian_Filter'.
[3] 'ksize' means the kernel size of the filter.
"""

import torch.nn as nn
import math
import kornia


class denoising_block(nn.Module):
    def __init__(self, in_planes, ksize, filter_type):
        super(denoising_block, self).__init__()
        self.in_planes = in_planes
        self.ksize = ksize
        self.filter_type = filter_type
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        if self.filter_type == 'Median_Filter':
            x_denoised = kornia.median_blur(x, (self.ksize, self.ksize))
        elif self.filter_type == 'Mean_Filter':
            x_denoised = kornia.box_blur(x, (self.ksize, self.ksize))
        elif self.filter_type == 'Gaussian_Filter':
            x_denoised = kornia.gaussian_blur2d(x, (self.ksize, self.ksize), (
            0.3 * ((x.shape[3] - 1) * 0.5 - 1) + 0.8, 0.3 * ((x.shape[2] - 1) * 0.5 - 1) + 0.8))
        new_x = x + self.conv(x_denoised)
        return new_x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, retain_activation=True, activation='ReLU'):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        if retain_activation:
            if activation == 'ReLU':
                self.block.add_module("ReLU", nn.ReLU(inplace=True))
            elif activation == 'LeakyReLU':
                self.block.add_module("LeakyReLU", nn.LeakyReLU(0.1))
            elif activation == 'Softplus':
                self.block.add_module("Softplus", nn.Softplus())
        self.block.add_module("MaxPool2d", nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

    def forward(self, x):
        out = self.block(x)
        return out


# Embedding network used in Matching Networks (Vinyals et al., NIPS 2016), Meta-LSTM (Ravi & Larochelle, ICLR 2017),
# MAML (w/ h_dim=z_dim=32) (Finn et al., ICML 2017), Prototypical Networks (Snell et al. NIPS 2017).

# origional size
# x_dim=3, h_dim=64, z_dim=64

# , whether_denoising=False, filter_type="Mean_Filter", ksize=3

class ProtoNetEmbedding(nn.Module):
    def __init__(self, x_dim=3, h_dim=64, z_dim=64, retain_last_activation=True, activation='ReLU',
                 whether_denoising=False, filter_type="Gaussian_Filter", ksize=3):
        super(ProtoNetEmbedding, self).__init__()
        self.whether_denoising = whether_denoising
        if whether_denoising:
            self.denoising_block1 = denoising_block(in_planes=3, ksize=ksize, filter_type=filter_type)
            self.denoising_block2 = denoising_block(in_planes=64, ksize=ksize, filter_type=filter_type)

        self.encoder = nn.Sequential(
            ConvBlock(x_dim, h_dim, activation=activation),
            ConvBlock(h_dim, h_dim, activation=activation),
            ConvBlock(h_dim, h_dim, activation=activation),
            ConvBlock(h_dim, z_dim, retain_activation=retain_last_activation, activation=activation),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # print('\n x.shape', x.shape)
        if self.whether_denoising:
            x = self.denoising_block1(x)
            # print('\n x denoise shape', x.shape)
        x = self.encoder(x)
        if self.whether_denoising:
            x = self.denoising_block2(x)
        return x.view(x.size(0), -1)
