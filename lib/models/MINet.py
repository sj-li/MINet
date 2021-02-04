import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init



class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.se = nn.Sequential(
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


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

class ClassifierModule(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(ClassifierModule, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding,
                          dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out



class Encoder(nn.Module):
    def __init__(self, c_in=5):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(c_in, 16, kernel_size=3, stride=(1, 2), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.layer1 = nn.Sequential(
                        Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
                        Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, (1, 2)),
                )

        self.layer2 = nn.Sequential(
                        Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
                        Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),
                )

        self.layer3 = nn.Sequential(
                        Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
                        Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
                        Block(3, 40, 240, 80, hswish(), None, 2),
                )

        self.layer4 = nn.Sequential(
                    Block(3, 80, 200, 80, hswish(), None, 1),
                    Block(3, 80, 184, 80, hswish(), None, 1),
                    Block(3, 80, 184, 80, hswish(), None, 1),
        )


    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out1 = out = self.layer1(out)
        out2 = out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out, out2, out1


class MINet(nn.Module):
    def __init__(self, c_in, num_classes, num_classes_moving, num_classes_static, dropout, use_mps=True, is_train=True):
        super(MINet, self).__init__()

        self.encoder_c = Encoder(c_in)
        self.encoder_diff = Encoder(c_in)

        self.fusion = nn.Sequential(
                            Block(5, 160, 672, 160, hswish(), SeModule(160), 1),
                            Block(5, 160, 672, 160, hswish(), SeModule(160), 1),
                            Block(5, 160, 960, 40, hswish(), SeModule(40), 1),
                        )

        self.decoder_static2 = Block(5, 40, 360, 24, hswish(), SeModule(24), 1)
        self.decoder_static1 = Block(5, 24, 180, 32, hswish(), SeModule(32), 1)

        self.decoder_moving2 = Block(5, 40, 360, 24, hswish(), SeModule(24), 1)
        self.decoder_moving1 = Block(5, 24, 180, 32, hswish(), SeModule(32), 1)

        self.classifier = ClassifierModule(64, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

        self.dp = nn.Dropout2d(p=dropout)
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1)

        self.init_params()

        self.conv_moving_e = nn.Conv2d(80, num_classes_moving, 3, padding=1)
        self.conv_moving_d = nn.Conv2d(32, num_classes_moving, 3, padding=1)

        self.conv_static_e = nn.Conv2d(80, num_classes_static, 3, padding=1)
        self.conv_static_d = nn.Conv2d(32, num_classes_static, 3, padding=1)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, 5, -1, h, w)
        x_diff = x[:,0,:,:,:]
        x_c = x[:,:,-1,:,:]

        sup_edge = []
        sup_moving = []
        sup_static = []

        x_c, x_c2, x_c1 = self.encoder_c(x_c)
        x_diff, x_diff2, x_diff1 = self.encoder_diff(x_diff)

        sup_static.append(self.conv_static_e(x_c))
        sup_moving.append(self.conv_moving_e(x_diff))

        x = self.fusion(torch.cat([x_c, x_diff], 1))

        x_static = self.decoder_static2(F.interpolate(x, size=(x_c2.size(2), x_c2.size(3)), mode='bilinear', align_corners=False) + x_c2)
        x_static = self.decoder_static1(F.interpolate(x_static, size=(x_c1.size(2), x_c1.size(3)), mode='bilinear', align_corners=False) + x_c1)

        x_moving = self.decoder_moving2(F.interpolate(x, size=(x_diff2.size(2), x_diff2.size(3)), mode='bilinear', align_corners=False) + x_diff2)
        x_moving = self.decoder_moving1(F.interpolate(x_moving, size=(x_diff1.size(2), x_diff1.size(3)), mode='bilinear', align_corners=False) + x_diff1)

        sup_static.append(self.conv_static_d(x_static))
        sup_moving.append(self.conv_moving_d(x_moving))

        #x = self.dp(x)
        x = self.classifier(torch.cat([x_static, x_moving], 1))

        x = F.interpolate(
            x, size=(x.size(2), x.size(3)*4), mode='bilinear', align_corners=False)

        x = F.softmax(x, dim=1)

        skips = {'sup_edge': sup_edge, 'sup_static': sup_static, 'sup_moving': sup_moving}

        return x, skips

