import torch.nn as nn
from torch.nn.utils.spectral_norm import spectral_norm as sn

from architecture.self_attention import NonLocal
from architecture.init_weight import init_func

nf = 64
out_scale = 10

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # prev conv
        self.pre_conv = nn.Sequential(
            sn(nn.Conv2d(3, 1 * nf, 3, 1, 1)),
            nn.ReLU(),
            sn(nn.Conv2d(1 * nf, 1 * nf, 3, 1, 1)),
            nn.AvgPool2d(2)
        )
        self.pre_skip = nn.Sequential(
            nn.AvgPool2d(2),
            sn(nn.Conv2d(3, 1 * nf, 1, 1, 0)),
        )
        # body
        self.body = nn.Sequential(
            ResBlock(1 * nf, 2 * nf, downsample=True),
            NonLocal(2 * nf),
            ResBlock(2 * nf, 4 * nf, downsample=True),
            ResBlock(4 * nf, 4 * nf, downsample=False),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.linear = sn(nn.Linear(4 * nf, 1))
        # embedding
        self.embed = nn.Embedding(1000, 4 * nf)
        self.embed = sn(self.embed)

        self.apply(init_func)

    def forward(self, input_x, class_id):
        # prev conv
        out = self.pre_conv(input_x)
        out += self.pre_skip(input_x)
        # body
        out = self.body(out)
        out = out.view(-1, 4 * nf)
        fc = self.linear(out).squeeze(1)
        embed = self.embed(class_id)
        prod = (out * embed).sum(1)
        return (fc + prod) * out_scale


class ResBlock(nn.Module):
    def __init__(self, fin, fout, downsample):
        super(ResBlock, self).__init__()
        self.learned_shortcut = (fin != fout) or downsample
        self.downsample = downsample
        self.conv_0 = sn(nn.Conv2d(fin, fout, 3, 1, 1))
        self.conv_1 = sn(nn.Conv2d(fout, fout, 3, 1, 1))
        if self.learned_shortcut:
            self.conv_s = sn(nn.Conv2d(fin, fout, 1, 1, 0))
        self.relu_0 = nn.ReLU(False)
        self.relu_1 = nn.ReLU(False)
        if downsample:
            self.down_0 = nn.AvgPool2d(2)
            self.down_s = nn.AvgPool2d(2)

    def forward(self, input_x):
        skip = self._shortcut(input_x)
        out = self.conv_0(self.relu_0(input_x))
        out = self.conv_1(self.relu_1(out))
        if self.downsample:
            out = self.down_0(out)
            skip = self.down_s(skip)
        out += skip
        return out

    def _shortcut(self, input_x):
        if self.learned_shortcut:
            return self.conv_s(input_x)
        else:
            return input_x
