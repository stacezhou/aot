import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.layers.basic import ConvGN
from .pa_module import PA_module
from mmcv.runner import load_checkpoint


class FPNSegmentationHead(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 decode_intermediate_input=True,
                 hidden_dim=256,
                 shortcut_dims=[24, 32, 96, 1280],
                 with_pa = True,
                 pa_pretrained = None,
                 align_corners=True):
        super().__init__()
        self.align_corners = align_corners

        self.decode_intermediate_input = decode_intermediate_input

        self.conv_in = ConvGN(in_dim, hidden_dim, 1)

        self.conv_16x = ConvGN(hidden_dim, hidden_dim, 3)
        self.conv_8x = ConvGN(hidden_dim, hidden_dim // 2, 3)
        self.conv_4x = ConvGN(hidden_dim // 2, hidden_dim // 2, 3)

        self.adapter_16x = nn.Conv2d(shortcut_dims[-2], hidden_dim, 1)
        self.adapter_8x = nn.Conv2d(shortcut_dims[-3], hidden_dim, 1)
        if with_pa:
            self.adapter_4x = nn.Conv2d(shortcut_dims[-4] + 256, hidden_dim // 2, 1)
        else:
            self.adapter_4x = nn.Conv2d(shortcut_dims[-4], hidden_dim // 2, 1)

        self.conv_out = nn.Conv2d(hidden_dim // 2, out_dim, 1)

        self._init_weight()
        if with_pa:
            self.pa = PA_module()
            if pa_pretrained is not None:
                load_checkpoint(self.pa, pa_pretrained)

        self.with_pa = with_pa

    def forward(self, inputs, shortcuts, imgs = None):

        if self.decode_intermediate_input:
            x = torch.cat(inputs, dim=1)
        else:
            x = inputs[-1]

        x = F.relu_(self.conv_in(x))
        x = F.relu_(self.conv_16x(self.adapter_16x(shortcuts[-2]) + x))

        x = F.interpolate(x,
                          size=shortcuts[-3].size()[-2:],
                          mode="bilinear",
                          align_corners=self.align_corners)
        x = F.relu_(self.conv_8x(self.adapter_8x(shortcuts[-3]) + x))

        x = F.interpolate(x,
                          size=shortcuts[-4].size()[-2:],
                          mode="bilinear",
                          align_corners=self.align_corners)

        if self.with_pa:
            shortcuts_4 = torch.cat([ shortcuts[-4], self.pa(imgs) ], dim=1)
        else:
            shortcuts_4 = shortcuts[-4]
        x = F.relu_(self.conv_4x(self.adapter_4x(shortcuts_4) + x))

        x = self.conv_out(x)

        return x

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
