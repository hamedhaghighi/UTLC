import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from model.common import conv, ResAttModuleDownUpPlus


class _ResGroup(nn.Module):
    def __init__(self, n_feats, is_nonlocal=False, use_checkpoint=True):
        super().__init__()
        self.block = ResAttModuleDownUpPlus(n_feats, is_nonlocal, use_checkpoint)
        self.tail = conv(n_feats, n_feats)

    def forward(self, x):
        return self.tail(self.block(x))


class RNAN(nn.Module):
    def __init__(self, args):
        super().__init__()
        n_resgroup = args.n_resgroups
        n_feats = args.n_feats
        colors = 3
        out_dim = 120  # MoL
        use_checkpoint = args.checkpoint
        self.stages = args.stages
        # input = lr + masked_hr + mask [+ vq_latent]
        self.head = conv(colors * 2 + 1, n_feats)
        self.fuse = conv(n_feats * 2, n_feats, kernel_size=1, bias=False)
        self.stage_embedding = nn.Embedding(args.stages, n_feats)
        self.body_nl_low = _ResGroup(n_feats, is_nonlocal=True, use_checkpoint=use_checkpoint)
        self.body = nn.ModuleList([_ResGroup(n_feats) for _ in range(n_resgroup - 2)])
        self.body_tail = conv(n_feats, n_feats)
        self.body_nl_high = _ResGroup(n_feats, is_nonlocal=True, use_checkpoint=use_checkpoint)
        self.tail = conv(n_feats, out_dim)
        if args.position_encoding:
            self.position_encoding = nn.Parameter(torch.zeros(args.stages, n_feats))
        else:
            self.position_encoding = None

    def apply_position_encoding(self, x):
        if self.position_encoding is not None:
            B, _, H, W = x.size()
            num_patches = H * W // self.stages
            position_encoding = self.position_encoding.view(1, -1, 1).expand(B, -1, num_patches)
            kernel_size = int(np.sqrt(self.stages))
            position_encoding = F.fold(position_encoding, kernel_size=kernel_size, output_size=H, stride=kernel_size)
            return x + position_encoding
        return x

    def forward(self, x, stage, prev_hidden):
        feats_shallow = self.head(x)
        feats_shallow = self.apply_position_encoding(feats_shallow)
        feats_shallow = self.fuse(torch.cat([feats_shallow, prev_hidden], dim=1))
        feats_shallow = feats_shallow + self.stage_embedding(stage)[:, :, None, None]
        res = self.body_nl_low(feats_shallow)
        for b in self.body:
            res = b(res)
        res = self.body_tail(res)
        res = self.body_nl_high(res)
        res_main = self.tail(res)
        return res_main, res
