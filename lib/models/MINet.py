import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, padding=1, bias=False)
        # self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
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

class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class MobileMMA(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_size, out_size):
        super(MobileMMA, self).__init__()
        self.nolinear = nn.ReLU(inplace=True)
        kernel_size = 3
        expand_size = 4*in_size

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=in_size, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.conv2 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_size)

    def forward(self, x):
        out = self.nolinear(self.bn1(self.conv1(x)))
        out = self.nolinear(self.bn2(self.conv2(out)))
        return out


class MobileBlock(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_size, out_size, stride=1, semodule=None):
        super(MobileBlock, self).__init__()
        self.stride = stride
        self.se = semodule
        self.nolinear = nn.ReLU(inplace=True)
        expand_size = 2*in_size
        kernel_size = 3

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=self.stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear(self.bn1(self.conv1(x)))
        out = self.nolinear(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class sSE(nn.Module):
    def __init__(self, out_channels):
        super(sSE, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(out_channels, 1, 1, 1, 0),
                                  nn.BatchNorm2d(1))
    def forward(self,x):
        x=self.conv(x)
        x=torch.sigmoid(x)
        return x

class cSE(nn.Module):
    def __init__(self, out_channels):
        super(cSE, self).__init__()
        self.conv1 = nn.Conv2d(out_channels, out_channels*2, 1)
        self.conv2 = nn.Conv2d(out_channels*2, out_channels, 1)
    def forward(self,x):
        x=nn.AvgPool2d(x.size()[2:])(x)
        x=self.conv1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=torch.sigmoid(x)
        return x

class fSE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(fSE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1)
    def forward(self,x):
        x=nn.AvgPool2d(x.size()[2:])(x)
        x=self.conv1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=torch.sigmoid(x)
        return x

class FCM(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(FCM, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, channels, 1),
                                  nn.BatchNorm2d(channels))
        self.conv2 = nn.Sequential(nn.Conv2d(channels, out_channels, 1),
                                  nn.BatchNorm2d(out_channels))
        self.spatial_gate = sSE(out_channels)
        self.channel_gate = cSE(out_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x),inplace=True)
        x = F.relu(self.conv2(x),inplace=True)
        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)
        x = g1*x + g2*x
        return x

class MPS(nn.Module):
    def __init__(self, in_high, in_low, num_classes):
        super(MPS, self).__init__()
        self.conv_high = nn.Conv2d(in_high, num_classes, 1)
        self.conv_low = nn.Conv2d(in_low, num_classes, 1)

    def forward(self, x_high, x_low):
        x_high = self.conv_high(x_high)
        x_low = self.conv_low(x_low)

        return x_high, x_low


class CascadeFeatureFusion(nn.Module):
    """CFF Unit"""

    def __init__(self, low_channels, high_channels, out_channels, nclass, norm_layer=nn.BatchNorm2d, is_train=True):
        super(CascadeFeatureFusion, self).__init__()
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, 3, padding=2, dilation=2, bias=False),
            norm_layer(out_channels)
        )
        self.conv_high = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, 1, bias=False),
            norm_layer(out_channels)
        )

        self.is_train = is_train
        if is_train:
            self.conv_low_cls = nn.Conv2d(out_channels, 1, 1, bias=False)

    def forward(self, x_low, x_high):
        x_low = F.interpolate(x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True)
        x_low = self.conv_low(x_low)
        x_high = self.conv_high(x_high)
        x = x_low + x_high
        x = F.relu(x, inplace=True)

        if self.is_train:
            x_low_cls = self.conv_low_cls(x_low)
            return x, x_low_cls
        else:
            return x, None

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out



class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out




class MINet(nn.Module):
    def __init__(self, in_channels, num_classes, dropout, use_mps=True, is_train=True):
        super(MINet, self).__init__()

        self.in_channels = in_channels

        self.conv_in = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(1, 4, 3, padding=1),
                        nn.BatchNorm2d(4, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True),
                        )
                    for i in range(self.in_channels)
                    ]
                )

        self.down = nn.Sequential(
            Block(3, 20, 20, 20, nn.ReLU(inplace=True), None, 1),
            Block(3, 20, 64, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), None, 2),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), None, 1),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), None, 1),
            Block(3, 40, 240, 80, hswish(), None, 1),
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            nn.Conv2d(80, 32, 1))
 
        self.conv_t = MobileBlock(20, 32)

        self.layer_s_1 = MobileBlock(32, 64)
        self.layer_s_2 = MobileBlock(64, 128)
        self.layer_s_3 = MobileBlock(128, 128)

        self.layer_m_1 = nn.Sequential(
                                MobileBlock(32, 32),
                                MobileBlock(32, 64)
                            )
        self.layer_m_2 = nn.Sequential(
                                MobileBlock(64, 64),
                                MobileBlock(64, 128)
                            )
        self.layer_m_3 = MobileBlock(128, 128)


        self.layer_l_1 = self._make_layer(BasicBlock, 32, 64, 1, 1)
        self.layer_l_2 = self._make_layer(BasicBlock, 64, 128, 1, 1)
        self.layer_l_3 = self._make_layer(BasicBlock, 128, 128, 1, 1)


        self.conv_f = nn.Conv2d(384, 32, 1)
        self.cf = CascadeFeatureFusion(32, 32, 32, num_classes, is_train=is_train)

        self.dp = nn.Dropout2d(p=dropout)
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=3, stride=1, padding=1)

        self.pool2 = nn.AvgPool2d(2)
        self.pool4 = nn.AvgPool2d(4)

        self.use_mps = use_mps and is_train
        if self.use_mps:
            self.mps_1 = MPS(64, 64, num_classes)
            self.mps_2 = MPS(128, 128, num_classes)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)



    def forward(self, x):
        xs = []
        #for i in range(self.in_channels):
        #    xs.append(self.conv_in[i](x[i,:,:,:,:]))
 
        for i in range(self.in_channels):
            xs.append(self.conv_in[i](x[:,i,:,:].unsqueeze(1)))
        xs = torch.cat(xs, 1)


        x = self.down(xs)

        h, w = x.size(2), x.size(3)

        edge = []
        seg = []

        x_s = self.layer_s_1(x)
        x_m = self.layer_m_1(self.pool2(x))
        x_l = self.layer_l_1(self.pool4(x))

        if self.use_mps:
            sups = self.mps_1(x_s, x_m)
            #TODO
            seg.append(sups[0])
            seg.append(sups[1])

        x_l = self.layer_l_2(x_l + self.pool2(x_m) + self.pool4(x_s))
        x_m = self.layer_m_2(x_m + self.pool2(x_s))
        x_s = self.layer_s_2(x_s)


        if self.use_mps:
            sups = self.mps_2(x_s, x_m)
            #TODO
            seg.append(sups[0])
            seg.append(sups[1])

        x_l = self.layer_l_3(x_l + self.pool2(x_m) + self.pool4(x_s))
        x_m = self.layer_m_3(x_m + self.pool2(x_s))
        x_s = self.layer_s_3(x_s)


       # Upsampling
        x_m = F.interpolate(
            x_m, size=(h, w), mode='bilinear', align_corners=False)
        x_l = F.interpolate(
            x_l, size=(h, w), mode='bilinear', align_corners=False)


        x = self.conv_f(torch.cat([x_s, x_m, x_l], 1))

        x, x_low_cls = self.cf(x, self.conv_t(xs))
        #TODO
        edge.append(x_low_cls)

        x = self.dp(x)
        x = self.classifier(x)
        x = F.softmax(x, dim=1)

        skips = {'edge': edge, 'seg': seg}

        return x, skips

