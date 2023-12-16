from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torch.nn import Module, Conv2d, Linear, MaxPool2d
import math
import torch.nn as nn
import copy
import torch
import torch.nn.functional as F


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class ResNetBase(nn.Module):
    def _decide_num_classes(self):
        if self.dataset == "cifar10" or self.dataset == "svhn":
            return 10
        elif self.dataset == "cifar100":
            return 100
        elif "imagenet" in self.dataset:
            return 1000
        elif "femnist" == self.dataset:
            return 62

    def _weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_block(
        self, block_fn, planes, block_num, stride=1, group_norm_num_groups=None
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_fn.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block_fn.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                norm2d(group_norm_num_groups, planes=planes * block_fn.expansion),
            )

        layers = []
        layers.append(
            block_fn(
                in_planes=self.inplanes,
                out_planes=planes,
                stride=stride,
                downsample=downsample,
                group_norm_num_groups=group_norm_num_groups,
            )
        )
        self.inplanes = planes * block_fn.expansion

        for _ in range(1, block_num):
            layers.append(
                block_fn(
                    in_planes=self.inplanes,
                    out_planes=planes,
                    group_norm_num_groups=group_norm_num_groups,
                )
            )
        return nn.Sequential(*layers)

    def train(self, mode=True):
        super(ResNetBase, self).train(mode)

        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.freeze_bn_affine:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False


def norm2d(group_norm_num_groups, planes):
    if group_norm_num_groups is not None and group_norm_num_groups > 0:
        # group_norm_num_groups == planes -> InstanceNorm
        # group_norm_num_groups == 1 -> LayerNorm
        return nn.GroupNorm(group_norm_num_groups, planes)
    else:
        return nn.BatchNorm2d(planes)


class Bottleneck(nn.Module):
    """
    [1 * 1, x]
    [3 * 3, x]
    [1 * 1, x * 4]
    """

    expansion = 4

    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        downsample=None,
        group_norm_num_groups=None,
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_planes, out_channels=out_planes, kernel_size=1, bias=False
        )
        self.bn1 = norm2d(group_norm_num_groups, planes=out_planes)

        self.conv2 = nn.Conv2d(
            in_channels=out_planes,
            out_channels=out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = norm2d(group_norm_num_groups, planes=out_planes)

        self.conv3 = nn.Conv2d(
            in_channels=out_planes,
            out_channels=out_planes * 4,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = norm2d(group_norm_num_groups, planes=out_planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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

        return out


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding."
    return nn.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )



class BasicBlock(nn.Module):
    """
    [3 * 3, 64]
    [3 * 3, 64]
    """

    expansion = 1

    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        downsample=None,
        group_norm_num_groups=None,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = norm2d(group_norm_num_groups, planes=out_planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = norm2d(group_norm_num_groups, planes=out_planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_cifar(ResNetBase):
    def __init__(
        self,
        resnet_size=8,
        scaling=4,
        save_activations=False,
        group_norm_num_groups=None,
        freeze_bn=False,
        freeze_bn_affine=False,
        num_classes=100
    ):
        super(ResNet_cifar, self).__init__()
        self.freeze_bn = freeze_bn
        self.freeze_bn_affine = freeze_bn_affine


        # define Model.
        if resnet_size % 6 != 2:
            raise ValueError("resnet_size must be 6n + 2:", resnet_size)
        block_nums = (resnet_size - 2) // 6
        block_fn = Bottleneck if resnet_size >= 44 else BasicBlock

        # decide the num of classes.
        self.num_classes = num_classes

        # add
        self.scaling = scaling
        self.expansion = block_fn.expansion

        # define layers.
        assert int(16 * scaling) > 0
        self.inplanes = int(16 * scaling)
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=(16 * scaling),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = norm2d(group_norm_num_groups, planes=int(16 * scaling))
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_block(
            block_fn=block_fn,
            planes=int(16 * scaling),
            block_num=block_nums,
            group_norm_num_groups=group_norm_num_groups,
        )
        self.layer2 = self._make_block(
            block_fn=block_fn,
            planes=int(32 * scaling),
            block_num=block_nums,
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
        )
        self.layer3 = self._make_block(
            block_fn=block_fn,
            planes=int(64 * scaling * 2),
            block_num=block_nums,
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
        )

        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.classifier = nn.Linear(
            in_features=int(64 * scaling * block_fn.expansion),
            out_features=self.num_classes,
        )

        # weight initialization based on layer type.
        self._weight_initialization()

        # a placeholder for activations in the intermediate layers.
        self.save_activations = save_activations
        self.activations = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print("conv1(bn1(relu)):", x.shape) # [32, 64, 32, 32]

        x = self.layer1(x)
        # print("layer1: ", x.shape) # [32, 64, 32, 32]
        
        activation1 = x
        x = self.layer2(x)
        # print("layer2: ", x.shape) # [32, 128, 16, 16]

        activation2 = x
        x = self.layer3(x)
        # print("layer3: ", x.shape) # [32, 512, 8, 8]

        activation3 = x
        x = self.avgpool(x)
        # print("avgpool: ", x.shape) # [32, 512, 1, 1]
        
        x = x.view(x.size(0),  -1)
        # print("x.view: ", x.shape) # [32, 512]

        feature = x
        y = self.classifier(x)
        # print("x_classifier: ", y.shape)

        if self.save_activations:
            self.activations = [activation1, activation2, activation3]

        return feature, y


class BasicBlock_ResLT(nn.Module):
    """
    [3 * 3, 64]
    [3 * 3, 64]
    """

    expansion = 1

    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        downsample=None,
        group_norm_num_groups=None,
    ):
        super(BasicBlock_ResLT, self).__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = norm2d(group_norm_num_groups, planes=out_planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = norm2d(group_norm_num_groups, planes=out_planes)

        self.downsample = downsample
        self.stride = stride

        # add
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = LambdaLayer(
                lambda x: F.pad(
                    x[:, :, ::2, ::2],
                    (0, 0, 0, 0, out_planes // 4, out_planes // 4),
                    "constant",
                    0,
                )
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # out += residual
        out += self.shortcut(x)
        out = self.relu(out)

        return out


class ResNet_cifar_ResLT(ResNetBase):
    def __init__(
        self,
        resnet_size=8,
        scaling=4,
        save_activations=False,
        group_norm_num_groups=None,
        freeze_bn=False,
        freeze_bn_affine=False,
        num_classes=10
    ):
        super(ResNet_cifar_ResLT, self).__init__()
        self.freeze_bn = freeze_bn
        self.freeze_bn_affine = freeze_bn_affine

        # define Model.
        if resnet_size % 6 != 2:
            raise ValueError("resnet_size must be 6n + 2:", resnet_size)
        block_nums = (resnet_size - 2) // 6
        block_fn = Bottleneck if resnet_size >= 44 else BasicBlock_ResLT

        # decide the num of classes.
        self.num_classes = num_classes

        # define layers.
        assert int(16 * scaling) > 0
        self.inplanes = int(16 * scaling)
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=(16 * scaling),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = norm2d(group_norm_num_groups, planes=int(16 * scaling))
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_block(
            block_fn=block_fn,
            planes=int(16 * scaling),
            block_num=block_nums,
            group_norm_num_groups=group_norm_num_groups,
        )
        self.layer2 = self._make_block(
            block_fn=block_fn,
            planes=int(32 * scaling),
            block_num=block_nums,
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
        )
        self.layer3 = self._make_block(
            block_fn=block_fn,
            planes=int(64 * scaling),
            block_num=block_nums,
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
        )

        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.classifier = nn.Linear(
            in_features=int(64 * scaling * block_fn.expansion),
            out_features=self.num_classes,
        )

        # weight initialization based on layer type.
        self._weight_initialization()

        # a placeholder for activations in the intermediate layers.
        self.save_activations = save_activations
        self.activations = None


        # add —�?1x1 conv can be replaced with more light-weight bn layer
        self.BNH = nn.BatchNorm2d(64 * scaling)
        self.BNM = nn.BatchNorm2d(64 * scaling)
        self.BNT = nn.BatchNorm2d(64 * scaling)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        activation1 = x
        x = self.layer2(x)
        activation2 = x
        x = self.layer3(x)
        activation3 = x

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        #
        # feature = x
        #
        # y = self.classifier(x)

        # add
        out = x
        head_fs, medium_fs, tail_fs = self.BNH(out), self.BNM(out), self.BNT(out)
        fs = torch.cat((head_fs, medium_fs, tail_fs), dim=0)

        feature = self.avgpool(fs).view(fs.size(0), -1)
        y = self.classifier(feature)

        c = y.size(0) // 3

        if self.save_activations:
            self.activations = [activation1, activation2, activation3]

        return feature, y, y[:c, :], y[c:c * 2, :], y[c * 2:, :]


if __name__ == '__main__':
    model = ResNet_cifar_ResLT(resnet_size=8, scaling=4, save_activations=False,
                         group_norm_num_groups=None, freeze_bn=False, freeze_bn_affine=False, num_classes=10)

