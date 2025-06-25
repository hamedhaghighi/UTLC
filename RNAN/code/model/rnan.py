import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.common import ResAttModuleDownUpPlus, conv


class _ResGroup(nn.Module):
    """A residual group block with attention and convolution."""

    def __init__(self, n_feats, is_nonlocal=False, use_checkpoint=True):
        super().__init__()
        # Residual attention module with optional non-local block
        self.block = ResAttModuleDownUpPlus(n_feats, is_nonlocal, use_checkpoint)
        # Tail convolution
        self.tail = conv(n_feats, n_feats)

    def forward(self, x):
        # Forward pass through the block and tail
        return self.tail(self.block(x))


class RNAN(nn.Module):
    """Main RNAN model for contextualized image restoration."""

    def __init__(self, args):
        super().__init__()
        n_resgroup = args.n_resgroups
        n_feats = args.n_feats
        colors = 3
        out_dim = 120  # Mixture of Logistics (MoL) output dimension
        use_checkpoint = args.checkpoint
        self.stages = args.stages
        # Input: low-res + masked high-res + mask
        self.head = conv(colors * 2 + 1, n_feats)
        # Fuse features from shallow and previous hidden state
        self.fuse = conv(n_feats * 2, n_feats, kernel_size=1, bias=False)
        # Embedding for stage information
        self.stage_embedding = nn.Embedding(args.stages, n_feats)
        # Non-local block at low level
        self.body_nl_low = _ResGroup(
            n_feats, is_nonlocal=True, use_checkpoint=use_checkpoint
        )
        # Main body: sequence of residual groups
        self.body = nn.ModuleList([_ResGroup(n_feats) for _ in range(n_resgroup - 2)])
        # Tail convolution for the body
        self.body_tail = conv(n_feats, n_feats)
        # Non-local block at high level
        self.body_nl_high = _ResGroup(
            n_feats, is_nonlocal=True, use_checkpoint=use_checkpoint
        )
        # Output layer
        self.tail = conv(n_feats, out_dim)
        # Optional position encoding
        if args.position_encoding:
            self.position_encoding = nn.Parameter(torch.zeros(args.stages, n_feats))
        else:
            self.position_encoding = None

    def apply_position_encoding(self, x):
        # Add position encoding if enabled
        if self.position_encoding is not None:
            B, _, H, W = x.size()
            num_patches = H * W // self.stages
            position_encoding = self.position_encoding.view(1, -1, 1).expand(
                B, -1, num_patches
            )
            kernel_size = int(np.sqrt(self.stages))
            position_encoding = F.fold(
                position_encoding,
                kernel_size=kernel_size,
                output_size=H,
                stride=kernel_size,
            )
            return x + position_encoding
        return x

    def forward(self, x, stage, prev_hidden):
        # Shallow feature extraction
        feats_shallow = self.head(x)
        feats_shallow = self.apply_position_encoding(feats_shallow)
        # Fuse with previous hidden state
        feats_shallow = self.fuse(torch.cat([feats_shallow, prev_hidden], dim=1))
        # Add stage embedding
        feats_shallow = feats_shallow + self.stage_embedding(stage)[:, :, None, None]
        # Pass through non-local and residual groups
        res = self.body_nl_low(feats_shallow)
        for b in self.body:
            res = b(res)
        res = self.body_tail(res)
        res = self.body_nl_high(res)
        # Output main result and features
        res_main = self.tail(res)
        return res_main, res
