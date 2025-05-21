import torch
import torch.nn as nn
import torch.nn.functional as F

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Hardswish(nn.Module):  # export-friendly version of nn.Hardswish()
    @staticmethod
    def forward(x):
        # return x * F.hardsigmoid(x)  # for torchscript and CoreML
        return x * F.hardtanh(x + 3, 0., 6.) / 6.  # for torchscript, CoreML and ONNX


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    # slice concat conv
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        """ print("***********************")
        for f in x:
            print(f.shape) """
        return torch.cat(x, self.d)

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        try:
            self.act = Hardswish() if act else nn.Identity()
        except:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))
    
class YOLOPSeg(nn.Module):
    def __init__(self, num_classes=1):
        super(YOLOPSeg, self).__init__()
        
        # Encoder layers (backbone)
        self.focus = Focus(3, 32, 3)                 # Layer 0: Focus [3, 32, 3]
        self.conv1 = Conv(32, 64, 3, 2)              # Layer 1: Conv [32, 64, 3, 2]
        self.bottleneck1 = BottleneckCSP(64, 64, 1)  # Layer 2: BottleneckCSP [64, 64, 1]
        self.conv2 = Conv(64, 128, 3, 2)             # Layer 3: Conv [64, 128, 3, 2]
        self.bottleneck2 = BottleneckCSP(128, 128, 3) # Layer 4: BottleneckCSP [128, 128, 3]
        self.conv3 = Conv(128, 256, 3, 2)            # Layer 5: Conv [128, 256, 3, 2]
        self.bottleneck3 = BottleneckCSP(256, 256, 3) # Layer 6: BottleneckCSP [256, 256, 3]
        self.conv4 = Conv(256, 512, 3, 2)            # Layer 7: Conv [256, 512, 3, 2]
        self.spp = SPP(512, 512, [5, 9, 13])         # Layer 8: SPP [512, 512, [5, 9, 13]]
        self.bottleneck4 = BottleneckCSP(512, 512, 1, False) # Layer 9: BottleneckCSP [512, 512, 1, False]
        self.conv5 = Conv(512, 256, 1, 1)            # Layer 10: Conv [512, 256, 1, 1]
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest') # Layer 11: Upsample
        self.concat1 = Concat()                       # Layer 12: Concat
        self.bottleneck5 = BottleneckCSP(512, 256, 1, False) # Layer 13: BottleneckCSP [512, 256, 1, False]
        self.conv6 = Conv(256, 128, 1, 1)            # Layer 14: Conv [256, 128, 1, 1]
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest') # Layer 15: Upsample
        self.concat2 = Concat()                       # Layer 16: Concat
        
        # Segmentation Head (following the structure you provided)
        self.seg_conv1 = Conv(256, 128, 3, 1)        # Layer 25: Conv [256, 128, 3, 1]
        self.seg_upsample1 = nn.Upsample(scale_factor=2, mode='nearest') # Layer 26: Upsample
        self.seg_bottleneck1 = BottleneckCSP(128, 64, 1, False) # Layer 27: BottleneckCSP [128, 64, 1, False]
        self.seg_conv2 = Conv(64, 32, 3, 1)          # Layer 28: Conv [64, 32, 3, 1]
        self.seg_upsample2 = nn.Upsample(scale_factor=2, mode='nearest') # Layer 29: Upsample
        self.seg_conv3 = Conv(32, 16, 3, 1)          # Layer 30: Conv [32, 16, 3, 1]
        self.seg_bottleneck2 = BottleneckCSP(16, 8, 1, False) # Layer 31: BottleneckCSP [16, 8, 1, False]
        self.seg_upsample3 = nn.Upsample(scale_factor=2, mode='nearest') # Layer 32: Upsample
        self.seg_conv4 = Conv(8, num_classes, 3, 1, act=False) # Layer 33: Conv [8, 2, 3, 1] (modified for multi-class)

    def forward(self, x):
        # Backbone forward pass with saved intermediate outputs for skip connections
        x0 = self.focus(x)                          # Layer 0
        x1 = self.conv1(x0)                          # Layer 1
        x2 = self.bottleneck1(x1)                    # Layer 2
        x3 = self.conv2(x2)                          # Layer 3
        x4 = self.bottleneck2(x3)                    # Layer 4
        x5 = self.conv3(x4)                          # Layer 5
        x6 = self.bottleneck3(x5)                    # Layer 6
        x7 = self.conv4(x6)                          # Layer 7
        x8 = self.spp(x7)                            # Layer 8
        x9 = self.bottleneck4(x8)                    # Layer 9
        x10 = self.conv5(x9)                         # Layer 10
        x11 = self.upsample1(x10)                    # Layer 11
        x12 = self.concat1([x11, x6])                # Layer 12 (concat with layer 6)
        x13 = self.bottleneck5(x12)                  # Layer 13
        x14 = self.conv6(x13)                        # Layer 14
        x15 = self.upsample2(x14)                    # Layer 15
        x16 = self.concat2([x15, x4])                # Layer 16 (concat with layer 4)

        # Segmentation head
        s = self.seg_conv1(x16)                     # [256, 128, 3, 1]
        s = self.seg_upsample1(s)                   # Upsample 2x
        s = self.seg_bottleneck1(s)                 # BottleneckCSP [128, 64, 1, False]
        s = self.seg_conv2(s)                       # [64, 32, 3, 1]
        s = self.seg_upsample2(s)                   # Upsample 2x
        s = self.seg_conv3(s)                       # [32, 16, 3, 1]
        s = self.seg_bottleneck2(s)                 # BottleneckCSP [16, 8, 1, False]
        s = self.seg_upsample3(s)                   # Upsample 2x
        s = self.seg_conv4(s)                       # Final output - modified for multi-class
        
        return s