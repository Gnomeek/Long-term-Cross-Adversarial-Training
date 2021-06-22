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
import torch.nn.functional as F
from models.dropblock import DropBlock
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


# This ResNet network was designed following the practice of the following papers:
# TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
# A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out
        # self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        # self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        # self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        # self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)


class ResNet(nn.Module):

    def __init__(self, block, keep_prob=1.0, avg_pool=False, drop_rate=0.0, dropblock_size=5, whether_denoising=False,
                 filter_type="Gaussian_Filter", ksize=3):
        self.inplanes = 3
        super(ResNet, self).__init__()
        self.whether_denoising = whether_denoising
        if whether_denoising:
            self.denoising_block1 = denoising_block(in_planes=3, ksize=ksize, filter_type=filter_type)
            self.denoising_block2 = denoising_block(in_planes=64, ksize=ksize, filter_type=filter_type)
        self.layer1 = self._make_layer(block, 8, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 16, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 32, stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size)
        if avg_pool:
            self.avgpool = nn.AvgPool2d(5, stride=1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # print('\n x.shape', x.shape)
        if self.whether_denoising:
            x = self.denoising_block1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.keep_avg_pool:
            x = self.avgpool(x)
        # print('\n x_out.shape', x.shape)
        if self.whether_denoising:
            x = self.denoising_block2(x)
        x = x.view(x.size(0), -1)
        # print('\n resnet x.shape return', x.shape)
        return x


def resnet12(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-12 model.
    """
    model = ResNet(BasicBlock, keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model
