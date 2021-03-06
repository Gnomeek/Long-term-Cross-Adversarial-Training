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
import torch
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


# Embedding network used in Meta-learning with differentiable closed-form solvers
# (Bertinetto et al., in submission to NIPS 2018).
# They call the ridge rigressor version as "Ridge Regression Differentiable Discriminator (R2D2)."

# Note that they use a peculiar ordering of functions, namely conv-BN-pooling-lrelu,
# as opposed to the conventional one (conv-BN-lrelu-pooling).

def R2D2_conv_block(in_channels, out_channels, retain_activation=True, keep_prob=1.0, activation='LeakyReLU'):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(2)
    )
    if retain_activation:
        if activation == 'LeakyReLU':
            block.add_module("LeakyReLU", nn.LeakyReLU(0.1))
        elif activation == 'ReLU':
            block.add_module("ReLU", nn.ReLU())
        elif activation == 'Softplus':
            block.add_module("Softplus", nn.Softplus())

    if keep_prob < 1.0:
        block.add_module("Dropout", nn.Dropout(p=1 - keep_prob, inplace=False))

    return block


# x_dim=3, h1_dim=96, h2_dim=192, h3_dim=384, z_dim=512
class R2D2Embedding(nn.Module):
    def __init__(self, x_dim=3, h1_dim=96, h2_dim=64, h3_dim=16, z_dim=16, \
                 retain_last_activation=False, denoise=False, activation='LeakyReLU', whether_denoising=False,
                 filter_type="Gaussian_Filter", ksize=3):
        super(R2D2Embedding, self).__init__()
        self.whether_denoising = whether_denoising
        if whether_denoising:
            self.denoising_block1 = denoising_block(in_planes=3, ksize=ksize, filter_type=filter_type)
            self.denoising_block2 = denoising_block(in_planes=16, ksize=ksize, filter_type=filter_type)

        self.block1 = R2D2_conv_block(x_dim, h1_dim, activation=activation)
        self.block2 = R2D2_conv_block(h1_dim, h2_dim, activation=activation)
        self.block3 = R2D2_conv_block(h2_dim, h3_dim, keep_prob=0.9, activation=activation)
        self.denoise = denoise
        # In the last conv block, we disable activation function to boost the classification accuracy.
        # This trick was proposed by Gidaris et al. (CVPR 2018).
        # With this trick, the accuracy goes up from 50% to 51%.
        # Although the authors of R2D2 did not mention this trick in the paper,
        # we were unable to reproduce the result of Bertinetto et al. without resorting to this trick.
        self.block4 = R2D2_conv_block(h3_dim, z_dim, retain_activation=retain_last_activation, keep_prob=0.7)

    def forward(self, x):
        # print('\n x.shape', x.shape)
        if self.whether_denoising:
            x = self.denoising_block1(x)
            # print('\n x denoise shape', x.shape)
        b1 = self.block1(x)
        b2 = self.block2(b1)
        if self.denoise:
            # print("before denoise", b2.size())
            _, n_in, H, W = b2.size()
            theta = nn.Conv2d(n_in, int(n_in / 2), 1,
                              stride=1, bias=False).to('cuda')
            phi = nn.Conv2d(n_in, int(n_in / 2), 1,
                            stride=1, bias=False).to('cuda')
            g = b2
            f = torch.einsum('niab,nicd->nabcd', theta(b2), phi(b2))
            orig_shape = f.size()
            f = torch.reshape(f, (-1, H * W, H * W))
            f = f / math.sqrt(n_in)
            softmax = torch.nn.Softmax(dim=0)
            f = softmax(f)
            f = torch.reshape(f, orig_shape)
            f = torch.einsum('nabcd,nicd->niab', f, g)
            final_conv = nn.Conv2d(f.size()[1], f.size()[1], 1, stride=1, bias=False).to('cuda')
            f = final_conv(f)
            b2 = b2 + f
            # print("after denoise", b2.size())
        b3 = self.block3(b2)
        # print('\n b3.shape', b3.shape)
        if self.whether_denoising:
            b3 = self.denoising_block2(b3)
        b4 = self.block4(b3)
        # Flatten and concatenate the output of the 3rd and 4th conv blocks as proposed in R2D2 paper.
        # return torch.cat((b3.view(b3.size(0), -1), b4.view(b4.size(0), -1)), 1)
        return b3.view(b3.size(0), -1)
