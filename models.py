"""Real ResNeXt-50 (32x4d), DenseBlocks and a plain CNN baseline."""
import torch
from torch import nn
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights

MODEL_NAMES = ('plain_cnn', 'resnext50', 'resnext50_dense')

class DenseLayer(nn.Module):
    def __init__(self, channels, growth=32):
        super().__init__()
        self.layers = nn.Sequential(nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, 4*growth, 1, bias=False),
            nn.BatchNorm2d(4*growth), nn.ReLU(),
            nn.Conv2d(4*growth, growth, 3, padding=1, bias=False))
    def forward(self, x):
        return torch.cat((x, self.layers(x)), dim=1)

class DenseBlock(nn.Sequential):
    def __init__(self, channels, layers=4, growth=32):
        super().__init__(*[DenseLayer(channels+i*growth, growth) for i in range(layers)])
        self.out_channels = channels + layers*growth

class GestureModel(nn.Module):
    def __init__(self, name, num_classes, pretrained=False):
        super().__init__()
        if name not in MODEL_NAMES:
            raise ValueError(name)
        if name == 'plain_cnn':
            if pretrained:
                raise ValueError('The custom plain CNN has no pretrained weights.')
            blocks, cin = [], 3
            for cout in (64, 64, 128, 256, 512):
                blocks += [nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                           nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2)]
                cin = cout
            self.backbone = nn.Sequential(*blocks)
            channels = 512
        else:
            weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V1 if pretrained else None
            base = resnext50_32x4d(weights=weights)
            # Preserve all [3, 4, 6, 3] bottleneck stages and the standard stem.
            self.backbone = nn.Sequential(base.conv1, base.bn1, base.relu,
                base.maxpool, base.layer1, base.layer2, base.layer3, base.layer4)
            channels = 2048
        if name == 'resnext50_dense':
            first = DenseBlock(256)
            second = DenseBlock(256)
            # No extra spatial downsampling: a 64x64 input reaches a 2x2 map.
            self.neck = nn.Sequential(nn.Conv2d(channels, 256, 1, bias=False),
                first, nn.BatchNorm2d(first.out_channels), nn.ReLU(),
                nn.Conv2d(first.out_channels, 256, 1, bias=False), second)
            channels = second.out_channels
        else:
            self.neck = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(channels, num_classes))
    def forward(self, x):
        x = self.neck(self.backbone(x))
        return self.head(self.pool(x).flatten(1))
