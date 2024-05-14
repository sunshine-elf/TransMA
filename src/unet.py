import math
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Sequential(
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projection(x)

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x):
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Module):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.1, shape=(14, 14)):
        super().__init__()

        self.f1=nn.Linear(emb_size, expansion * emb_size)

        self.dw1=nn.Conv2d(in_channels=emb_size*expansion, out_channels=emb_size*expansion, kernel_size=5, groups=emb_size*expansion, stride=1, padding=2)
        self.gelu=nn.GELU()

        self.f2=nn.Linear(expansion * emb_size, emb_size)
        self.d1=nn.Dropout(drop_p)



    def forward(self, x,shape):

        x=self.f1(x)
        x=rearrange(x,'b (h w) e -> b e (h) (w)', h=shape[0])
        x=self.gelu(self.dw1(x))
        x=rearrange(x,'b e (h) (w) -> b (h w) e')
        x=self.f2(x)
        x=self.d1(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.1,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 num_heads=8
                 ):
        super().__init__()
        self.res1=ResidualAdd(nn.Sequential(
            nn.LayerNorm(emb_size),
            MultiHeadAttention(emb_size=emb_size, num_heads=num_heads, dropout=drop_p),
            nn.Dropout(drop_p)
        ))

        self.ln=nn.LayerNorm(emb_size)
        self.ffd=FeedForwardBlock(
                emb_size, expansion=forward_expansion, drop_p=forward_drop_p)
        self.d1=nn.Dropout(drop_p)

    def forward(self,data):
        x,shape=data
        x=self.res1(x)

        x=self.ln(x)
        x=self.ffd(x,shape)
        x=self.d1(x)
        return x,shape



class TransformerEncoder(nn.Module):

    def __init__(self,
                 depth=12,
                 emb_size: int = 768,
                 drop_p: float = 0.1,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ):
        super().__init__( )
        self.trans_blocks=nn.Sequential(*[TransformerEncoderBlock(emb_size=emb_size,drop_p=drop_p, forward_drop_p=forward_drop_p,
                                      forward_expansion=forward_expansion) for _ in range(depth)],)
    def forward(self,data):
        x,shape=data
        x,_=self.trans_blocks((x,shape))
        x=rearrange(x,'b (h w) e -> b e (h) (w)', h=shape[0])
        return x


class Transformer(nn.Module):
    def __init__(self, depth=12, emb_size=768):
        super().__init__()
        self.patch_embedding = PatchEmbedding()
        self.transformer_encoder = TransformerEncoder(depth=depth, emb_size=emb_size)

    def forward(self, x):
        shape = x.shape[-2:]
        x = self.patch_embedding(x)
        x = self.transformer_encoder((x, shape))
        return x


class DropBlock(nn.Module):
    def __init__(self, block_size: int = 5, p: float = 0.1):
        super().__init__()
        self.block_size = block_size
        self.p = p

    def calculate_gamma(self, x):
        """计算gamma
        Args:
            x (Tensor): 输入张量
        Returns:
            Tensor: gamma
        """

        invalid = (1 - self.p) / (self.block_size ** 2)
        valid = (x.shape[-1] ** 2) / ((x.shape[-1] - self.block_size + 1) ** 2)
        return invalid * valid

    def forward(self, x):
        N, C, H, W = x.size()
        if self.training:
            gamma = self.calculate_gamma(x)
            mask_shape = (N, C, H - self.block_size + 1, W - self.block_size + 1)
            mask = torch.bernoulli(torch.full(mask_shape, gamma, device=x.device))
            mask = F.pad(mask, [self.block_size // 2] * 4, value=0)
            mask_block = 1 - F.max_pool2d(
                mask,
                kernel_size=(self.block_size, self.block_size),
                stride=(1, 1),
                padding=(self.block_size // 2, self.block_size // 2),
            )
            x = mask_block * x * (mask_block.numel() / mask_block.sum())
        return x


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            DropBlock(7, 0.9),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            DropBlock(7, 0.9),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Attention(nn.Sequential):
    def __init__(self, in_channels,out_channels=None):

        super(Attention, self).__init__()
        self.out_ch=out_channels
        self.dropout = nn.Dropout(p=0.18)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.shared_conv = nn.Conv1d(1, 1, kernel_size=3, bias=False, padding=1)

        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        if out_channels:
            self.out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.max_pool(x)
        x2 = self.avg_pool(x)
        x1 = x1.view(x1.shape[0], 1, -1)
        x2 = x2.view(x2.shape[0], 1, -1)
        s1 = self.shared_conv(x1)
        s2 = self.shared_conv(x2)
        s1 = s1.view(x1.shape[0], -1, 1, 1)
        s2 = s2.view(x2.shape[0], -1, 1, 1)
        s = s1 + s2
        s = torch.sigmoid(s)
        c = x * s

        x3 = torch.cat((torch.max(x, 1, keepdim=True)[0], torch.min(x, 1, keepdim=True)[0]), dim=1)
        x4 = torch.sigmoid(self.conv(x3))
        x5 = x * x4

        x = c + x5
        if self.out_ch:
            x=self.out(x)
        return x


class MSR(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSR, self).__init__()
        self.depth_wise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                                    groups=in_channels)
        self.point_wise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # self.relu = nn.ReLU()
        self.BN = nn.BatchNorm2d(out_channels)
        self.dilation = 1

        self.atr = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=3,padding=3)

        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1,padding=1)
        self.acb1 = nn.Conv2d(in_channels, out_channels, (1, 3), padding=[0,1])
        self.acb2 = nn.Conv2d(in_channels, out_channels, (3, 1), padding=[1,0])

        self.att = Attention(in_channels,out_channels=out_channels)

    def forward(self, x):
        x_1 = self.depth_wise(x)
        x_2 = self.point_wise(x_1)
        x_3 = self.BN(F.relu(x_2))
        x_4 = self.depth_wise(x_3)
        x_5 = self.point_wise(x_4)
        x_6 = F.relu(x_5)

        s_1 = self.atr(x)
        s_2 = self.BN(F.relu(s_1))
        s_3 = self.atr(s_2)
        s_4 = self.BN(s_3)

        t_1 = self.conv(x)
        t_2 = self.BN(F.relu(t_1))
        t_3 = self.acb1(t_2)
        t_4 = self.BN(F.relu(t_2))
        t_5 = self.acb2(t_4)
        t_6 = self.BN(t_5)

        f = x_6 + s_4 + t_6

        f = self.att(f)
        x = f + x

        return x


class Last_Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Last_Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            DropBlock(7, 0.18),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Conv1(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Conv1, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            DropBlock(7, 0.18),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


#
# class Up(nn.Module):
#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super(Up, self).__init__()
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv = MSR(in_channels, out_channels)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels)
#
#     def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
#         x1 = self.up(x1)
#         print(x1.shape)
#         print(x2.shape)
#         # [N, C, H, W]
#         diff_y = x2.size()[2] - x1.size()[2]
#         diff_x = x2.size()[3] - x1.size()[3]
#
#         # padding_left, padding_right, padding_top, padding_bottom
#         x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
#                         diff_y // 2, diff_y - diff_y // 2])
#
#         x = torch.cat([x2, x1], dim=1)
#         x = self.conv(x)
#         return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.i = in_channels
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, bias=False, dilation=3),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=3, bias=False, dilation=3),
            nn.BatchNorm2d(out_channels),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),

            nn.Conv2d(out_channels, out_channels, kernel_size=[1, 3], padding=[0, 1], bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),

            nn.Conv2d(out_channels, out_channels, kernel_size=[3, 1], padding=[1, 0], bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=[1, 1], padding=[0, 0], bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
        self.drop = DropBlock()
        self.attention = Attention(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:

        x1 = self.up(x1)

        x = torch.cat([x2, x1], dim=1)

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        res = branch1 + branch2 + branch3
        res = self.drop(res)

        res = self.attention(res) + self.conv(x)

        res = self.relu(res)
        return res


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)

        self.linear_projection = nn.Conv2d(in_channels=base_c * 8, out_channels=768, kernel_size=1, stride=1, padding=0)
        self.transformer = Transformer(depth=12, emb_size=768)
        self.linear_projection2 = nn.Conv2d(in_channels=768, out_channels=base_c*8, kernel_size=1, stride=1, padding=0)

        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

        self.msr5=MSR(in_channels=base_c*16//factor,out_channels=base_c*8)
        self.attention4=Attention(in_channels=base_c*8,out_channels=base_c*4)
        self.attention3=Attention(in_channels=base_c*4,out_channels=base_c*2)
        self.attention2 = Attention(in_channels=base_c * 2, out_channels=base_c)

        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        ## cnn
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        ##
        t4 = self.linear_projection(x4)
        t4 = self.transformer(t4)
        t4 = self.linear_projection2(t4)

        x5 = self.down4(t4)

        x5=self.msr5(x5)
        x4=x4
        x3=x3+F.interpolate(self.attention4(x4),size=x3.shape[-2:],mode='bilinear',align_corners=True)
        x2=x2+F.interpolate(self.attention3(x3),size=x2.shape[-2:],mode='bilinear',align_corners=True)
        x1 = x1 + F.interpolate(self.attention2(x2), size=x1.shape[-2:], mode='bilinear', align_corners=True)



        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return {"out": logits}


if __name__ == '__main__':
    model = UNet(in_channels=3, num_classes=1)
    inputdata = torch.randn(1, 3, 224, 224)
    out = model(inputdata)
    mask=torch.ones([1,1,224,224])
    loss=F.binary_cross_entropy(torch.sigmoid(out['out']),mask)
    loss.backward()
