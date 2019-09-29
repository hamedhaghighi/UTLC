import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


def conv(in_channels, out_channels, kernel_size=3, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class ResBlock(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.net = nn.Sequential(conv(n_feat, n_feat), nn.ReLU(True), conv(n_feat, n_feat))

    def forward(self, x):
        return self.net(x) + x


def _attention(*inputs):
    query, key, value = inputs
    logits = torch.matmul(query, key) / np.sqrt(key.size(1))
    spatial_attention = F.softmax(logits, dim=2)
    return torch.matmul(spatial_attention, value)


class NonLocalBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels, use_checkpoint):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.query = conv(self.in_channels, self.inter_channels, kernel_size=1, bias=False)
        self.key = conv(self.in_channels, self.inter_channels, kernel_size=1, bias=False)
        self.value = conv(self.in_channels, self.inter_channels, kernel_size=1, bias=False)
        self.W = conv(self.inter_channels, self.in_channels, kernel_size=1, bias=True)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        batch_size = x.size(0)
        value = self.value(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        query = self.query(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        key = self.key(x).view(batch_size, self.inter_channels, -1)
        if self.use_checkpoint and self.training:
            y = checkpoint.checkpoint(_attention, query, key, value)
        else:
            y = _attention(query, key, value)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        return self.W(y) + x


class MaskBranchDownUp(nn.Module):
    def __init__(self, n_feat, is_nonlocal=False, use_checkpoint=True):
        super().__init__()
        self.non_local = NonLocalBlock2D(n_feat, n_feat, use_checkpoint) if is_nonlocal else nn.Identity()
        self.head = ResBlock(n_feat)
        self.before_middle = nn.Conv2d(n_feat, n_feat, 3, stride=2, padding=1)
        self.middle = nn.ModuleList([ResBlock(n_feat) for _ in range(2)])
        self.after_middle = nn.ConvTranspose2d(n_feat, n_feat, 6, stride=2, padding=2)
        self.tail_rb = ResBlock(n_feat)
        self.tail = nn.Sequential(nn.Conv2d(n_feat, n_feat, 1, padding=0, bias=True), nn.Sigmoid())

    def forward(self, x):
        h = self.non_local(x)
        head = self.head(h)
        h = self.before_middle(head)
        for m in self.middle:
            h = m(h)
        h = self.after_middle(h)
        h = self.tail_rb(h + head)
        return self.tail(h)


class ResAttModuleDownUpPlus(nn.Module):
    def __init__(self, n_feat, is_nonlocal=False, use_checkpoint=True):
        super().__init__()
        self.head = nn.ModuleList([ResBlock(n_feat) for _ in range(1)])
        self.trunk = nn.ModuleList([ResBlock(n_feat) for _ in range(2)])  # t=2
        self.mask = MaskBranchDownUp(n_feat, is_nonlocal, use_checkpoint)
        self.tail = nn.ModuleList([ResBlock(n_feat) for _ in range(2)])  # q=2

    def forward(self, input):
        x = input
        for h in self.head:
            x = h(x)
        tx = x
        for t in self.trunk:
            tx = t(tx)
        mx = self.mask(x)
        y = tx * mx + x
        for t in self.tail:
            y = t(y)
        return y
