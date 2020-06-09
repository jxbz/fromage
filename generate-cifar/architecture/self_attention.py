import torch
from torch import nn
from torch.nn.utils.spectral_norm import spectral_norm as sn

class NonLocal(nn.Module):
    # Originally by https://github.com/zhaoyuzhi
    def __init__(self, in_dim):
        super(NonLocal, self).__init__()
        self.chanel_in = in_dim
        self.theta = sn(nn.Conv2d(in_dim, in_dim // 8, 1, 1, 0))
        self.phi = sn(nn.Conv2d(in_dim, in_dim // 8, 1, 1, 0))
        self.g = sn(nn.Conv2d(in_dim, in_dim // 2, 1, 1, 0))
        self.out_conv = sn(nn.Conv2d(in_dim // 2, in_dim, 1, 1, 0))
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        n, c, h, w = x.size()
        theta = self.theta(x).view(n, -1, h * w).permute(0, 2, 1)

        phi = self.phi(x)
        phi = self.max_pool(phi).view(n, -1, h * w // 4)

        energy = torch.bmm(theta, phi)
        attention = self.softmax(energy)

        g = self.g(x)
        g = self.max_pool(g).view(n, -1, h * w // 4)

        out = torch.bmm(g, attention.permute(0, 2, 1))
        out = out.view(n, c // 2, h, w)
        out = self.out_conv(out)

        out = self.gamma * out + x
        return out
