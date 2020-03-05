import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm as sn

from architecture.self_attention import NonLocal
from architecture.init_weight import init_func

nf = 64

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.embedding_size = 80
        self.z_dim_chunk = 20
        self.num_chunk = 4
        self.cond_dim = self.z_dim_chunk + self.embedding_size

        self.fc = sn(nn.Linear(self.z_dim_chunk, 4 * 4 * 4 * nf))
        self.embedding = sn(nn.Linear(1000, self.embedding_size))

        self.resblock_0 = ResBlock(4 * nf, 4 * nf, cond_dim=self.cond_dim)
        self.resblock_1 = ResBlock(4 * nf, 2 * nf, cond_dim=self.cond_dim)
        self.non_local = NonLocal(2 * nf)
        self.resblock_2 = ResBlock(2 * nf, 1 * nf, cond_dim=self.cond_dim)

        self.bn_img = nn.BatchNorm2d(nf)
        self.relu_img = nn.ReLU(True)
        self.conv_img = sn(nn.Conv2d(nf, 3, 3, 1, 1))

        self.apply(init_func)


    def forward(self, z, label):
        bs = label.size(0)
        # sample random noise
        if z is None:
            z = torch.randn(
                bs, self.z_dim_chunk*self.num_chunk, dtype=torch.float32).cuda()
        # noise splitting
        zs = torch.split(
            z, [self.z_dim_chunk] * self.num_chunk, dim=1)
        # class embedding
        onehot_class = torch.zeros([bs, 1000]).cuda().scatter(
            1, label.view(-1, 1), 1)
        embed = self.embedding(onehot_class)
        cond1 = torch.cat([embed, zs[1]], dim=1)
        cond2 = torch.cat([embed, zs[2]], dim=1)
        cond3 = torch.cat([embed, zs[3]], dim=1)

        # network
        x = self.fc(zs[0])
        x = x.view(bs, 4 * nf, 4, 4)   # 4x4
        
        x = self.resblock_0(x, cond1)  # 8x8
        x = self.resblock_1(x, cond2)  # 16x16
        x = self.non_local(x)          # non-local
        x = self.resblock_2(x, cond3)  # 32x32

        x = self.conv_img(self.relu_img(self.bn_img(x)))
        return torch.tanh(x)

class CondBN(nn.Module):
    def __init__(self, nc, cond_dim):
        super().__init__()
        self.mlp_gamma = sn(nn.Linear(cond_dim, nc))
        self.mlp_beta = sn(nn.Linear(cond_dim, nc))
        self.batch_norm = nn.BatchNorm2d(nc, momentum=0.001, affine=False)

    def forward(self, x, cond):
        n, c, _, _ = x.size()
        gamma = self.mlp_gamma(cond).view(n, c, 1, 1)
        beta = self.mlp_beta(cond).view(n, c, 1, 1)
        normalized = self.batch_norm(x)
        return normalized * (1.0 + gamma) + beta

class ResBlock(nn.Module):
    def __init__(self, fin, fout, cond_dim):
        super(ResBlock, self).__init__()
        self.learned_shortcut = (fin != fout)
        self.conv_0 = sn(nn.Conv2d(fin, fout, 3, 1, 1))
        self.conv_1 = sn(nn.Conv2d(fout, fout, 3, 1, 1))
        if self.learned_shortcut:
            self.conv_s = sn(nn.Conv2d(fin, fout, 1, 1, 0))
        self.relu_0 = nn.ReLU(True)
        self.relu_1 = nn.ReLU(True)
        self.cbn_0 = CondBN(fin, cond_dim)
        self.cbn_1 = CondBN(fout, cond_dim)

    def forward(self, input_x, cond):
        skip = self._shortcut(input_x, cond)
        out = self.relu_0(self.cbn_0(input_x, cond))
        out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.conv_0(out)
        out = self.conv_1(self.relu_1(self.cbn_1(out, cond)))
        out += skip
        return out

    def _shortcut(self, input_x, cond):
        skip = F.interpolate(input_x, scale_factor=2, mode='nearest')
        if self.learned_shortcut:
            return self.conv_s(skip)
        else:
            return skip