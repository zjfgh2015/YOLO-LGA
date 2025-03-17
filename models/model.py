import argparse
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, profile, scale_img, select_device,
                               time_sync)


class LGA_Block(nn.Module):

    def __init__(self, c_in, dim, input_resolution, num_heads=4, save=False, window_size=10,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()

        self.c_in = c_in
        self.dim = dim
        self.input_resolution = to_2tuple(input_resolution)
        self.conv_z1 = Conv(c1=c_in, c2=dim, k=3, s=1)
        self.conv_x = Conv(c1=c_in, c2=dim, k=3, s=1)
        self.c3_block = C3(c1=dim, c2=dim)
        self.aggregate = Conv(dim * 2, dim * 2, 3, 1, act=False)
        self.save = save

        self.local2global = CrossWindowAttention(dim=self.dim, input_resolution=self.input_resolution,
                                                 num_heads=num_heads, window_size=window_size,
                                                 mlp_ratio=mlp_ratio,
                                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                 drop=drop, attn_drop=attn_drop,
                                                 drop_path=drop_path,
                                                 norm_layer=norm_layer)

        self.global_block = SwinTransformerBlock(dim=self.dim, input_resolution=self.input_resolution,
                                                 num_heads=num_heads, window_size=window_size,
                                                 shift_size=window_size // 2,
                                                 mlp_ratio=mlp_ratio,
                                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                 drop=drop, attn_drop=attn_drop,
                                                 drop_path=drop_path,
                                                 norm_layer=norm_layer)

    def forward(self, x):
        """
       Args:
            x: input features with shape of (batch_size, C,H,W)
       return:
            z:output features with shape of (batch_size, C,H,W)
        """
        z = self.conv_z1(x)
        B, C, H, W = z.shape
        z = z.permute(0, 2, 3, 1).contiguous()
        z = z.view(B, H * W, C)
        z = self.global_block(z=z, x=z)  # B, H * W, C

        x = self.conv_x(x)
        x = self.c3_block(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, H * W, C)
        x = self.local2global(z=x, x=z)

        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        z = z.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = self.aggregate(torch.cat([x, z], 1))
        return x


class Backbone(nn.Module):
    def __init__(self, d, dpr):
        super().__init__()
        self.layers = nn.ModuleList()
        index = 0
        for i, (m, args) in enumerate(d):  # module, args
            if m == "Conv":
                m_ = Conv(*args)
            elif m == "C3":
                m_ = C3(*args)
            elif m == "LGA":
                m_ = LGA_Block(c_in=args[0], dim=args[1],
                               input_resolution=args[2],
                               num_heads=args[3], save=args[4],
                               drop_path=dpr[index])
                index += 1
            elif m == "SPPCSPC":
                m_ = SPPCSPC(*args)
            else:
                print(f'{m} is not considered in backbone,please check')
            self.layers.append(m_)

    def forward(self, x):
        save = []
        for layer in self.layers:
            x = layer(x)
            if hasattr(layer, 'save'):
                if layer.save:
                    save.append(x)
        save.append(x)
        return save


class Head(nn.Module):
    def __init__(self, d, dpr):
        super().__init__()
        self.conv_trans_layers = nn.ModuleList()
        self.reduce_layers = nn.ModuleList()
        self.downsample = nn.ModuleList()
        index = 0
        for i, (m, args) in enumerate(d):
            if m == "LGA":
                m_ = LGA_Block(c_in=args[0], dim=args[1],
                               input_resolution=args[2],
                               num_heads=args[3], save=args[4],
                               drop_path=dpr[index])
                index += 1
                self.conv_trans_layers.append(m_)
            elif m == "reduce":
                m_ = Conv(*args)
                self.reduce_layers.append(m_)
            elif m == "Conv":
                m_ = Conv(*args)
                self.downsample.append(m_)
            else:
                print(f'{m} is not considered in neck,please check')
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.concat = Concat(dimension=1)

    def forward(self, x):
        """
        input:
            z:input features with shape of bs,512,20,20
            x:input features with shape of bs,512,20,20
            save: list for save reauired features

        """
        result = []
        p1, p2, p3 = x[0], x[1], x[2]

        x1 = self.reduce_layers[0](p3)
        x = self.upsample(x1)
        x = self.concat([x, p2])
        x = self.reduce_layers[1](x)
        x = self.conv_trans_layers[0](x)

        x2 = self.reduce_layers[2](x)  # bs,256,40,40-->bs,128,40,40
        x = self.upsample(x2)
        x = self.concat([x, p1])
        x = self.reduce_layers[3](x)  # bs,256,80,80-->bs,128,80,80
        x = self.conv_trans_layers[1](x)
        result.append(x)  # save 80,80,128

        x = self.downsample[0](x)
        x = self.concat([x, x2])  # bs,256,40,40
        x = self.conv_trans_layers[2](x)
        result.append(x)  # save 40,40,256

        x = self.downsample[1](x)
        x = self.concat([x, x1])
        x = self.conv_trans_layers[3](x)
        result.append(x)
        return result
