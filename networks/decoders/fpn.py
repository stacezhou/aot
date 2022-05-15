import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.layers.basic import ConvGN
class DoubleConv(nn.Module):
    def __init__(self,in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            # nn.BatchNorm2d(out_ch), #添加了BN层
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.channel_up = nn.ModuleList([
            DoubleConv(in_ch,32),
            DoubleConv(32,64),
            DoubleConv(64,128),
            DoubleConv(128,256),
        ])

        self.size_down = nn.ModuleList([
            nn.MaxPool2d(2),
            nn.MaxPool2d(2),
            nn.MaxPool2d(2),
        ])
        
        self.channel_down = nn.ModuleList([
            DoubleConv(256,128),
            DoubleConv(128,64),
            DoubleConv(64,32),
            nn.Conv2d(32,in_ch,1),
        ])

        self.size_up = nn.ModuleList([
            nn.ConvTranspose2d(256, 128, 2 ,stride=2),
            nn.ConvTranspose2d(128, 64,  2 ,stride=2),
            nn.ConvTranspose2d(64,  32,  2 ,stride=2),
        ])
    
    def forward(self,x):
        shortcuts = []

        for i in range(3):
            x = self.channel_up[i](x)
            shortcuts.append(x)
            x = self.size_down[i](x)
        
        x = self.channel_up[3](x)

        for i in range(3):
            x = self.size_up[i](x)
            x = F.interpolate(x, size=shortcuts[-i-1].shape[-2:],
                        mode="bilinear", align_corners=True)
            x = torch.cat([x, shortcuts[-i-1]], dim=1)
            x = self.channel_down[i](x)
        
        x = self.channel_down[3](x)
        return x


class FPNSegmentationHead(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 decode_intermediate_input=True,
                 hidden_dim=256,
                 shortcut_dims=[24, 32, 96, 1280],
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
        self.adapter_4x = nn.Conv2d(shortcut_dims[-4], hidden_dim // 2, 1)

        self.conv_out = nn.Conv2d(hidden_dim // 2, out_dim, 1)
        self.unet = Unet(11,11)

        self._init_weight()

    def forward(self, inputs, shortcuts):

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
        x = F.relu_(self.conv_4x(self.adapter_4x(shortcuts[-4]) + x))

        x = self.conv_out(x)
        # TODOX 增加一个 channel 融合 / 抑制机制
        # TODOX channel 自适应关联性
        aux_x = self.unet(x)
        x = x + aux_x
        return x

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
