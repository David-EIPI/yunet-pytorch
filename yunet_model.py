"""
    Yunet model reproduced with Pytorch.
"""
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

class DWUnit(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        with_activation = True,
        device = 'cpu'
    ):
        super().__init__()
        # Pointwise 1x1 convolution. YuNet depthwise convolution first applies 1x1 pixel kernel, then 3x3 per-channel kernel
        self.conv1p = nn.Conv2d(in_channels, out_channels, 1, stride = 1, padding = 0, bias = True, groups = 1, device=device)
        self.conv1d = nn.Conv2d(out_channels, out_channels, 3, stride = 1, padding = 1, bias = True, groups = out_channels, device=device)
        if with_activation:
            self.bn = nn.BatchNorm2d(out_channels, device=device)
            self.act = nn.ReLU(inplace = True)
        else:
            self.act = None

    def forward(self, x):
        x = self.conv1p(x)
        x = self.conv1d(x)
        if self.act is not None:
            x = self.bn(x)
            x = self.act(x)
        return x


class DWBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        device='cpu'
    ):
        super().__init__()
        self.conv1 = DWUnit(in_channels, in_channels, device=device)
        self.conv2 = DWUnit(in_channels, out_channels, device=device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ConvHead(nn.Module):
    def __init__(
        self,
        color_channels,
        in_channels,
        out_channels,
        device = 'cpu'
    ):
        super().__init__()
        #print(device)
        self.conv1 = nn.Conv2d(color_channels, in_channels, 3, stride=2, padding=1, bias = True, groups=1, device=device)
        self.bn = nn.BatchNorm2d(in_channels, device=device)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = DWUnit(in_channels, out_channels, device=device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

class Backbone(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_hidden = 1,
        num_outputs = 3,
        device = 'cpu'
    ):
        super().__init__()
        # Backbone immediately follows the ConvHead block with the first MaxPool/2 downsampling layer
        self.hidden_blocks = nn.ModuleList(
            [
#                nn.MaxPool2d(2),
                DWBlock(in_channels, hidden_channels, device=device)
            ]
        )
        # More of the hidden DW blocks (1 in in the original layer)
        self.hidden_blocks.extend(
            [
                DWBlock(hidden_channels, hidden_channels, device=device) for i in range(0, num_hidden)
            ]
        )

        # The output blocks. Their output is fed to the TFPN (tiny feature pyramid network) layers.
        # The schematic in the original paper is misleading as it shows MaxPool layers after the corresponding DWBlock layers.
        # The code in the repository, as well as saved models have MaxPool/2 in the beginning of the output blocks.

        self.out_blocks = nn.ModuleList(
            [
                module for _ in range(0, num_outputs) for module in [nn.MaxPool2d(2), DWBlock(hidden_channels, hidden_channels, device=device)]
            ]
        )
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            if module.bias is not None:
                nn.init.xavier_normal_(module.weight.data)
                module.bias.data.fill_(0.02)
            else:
                module.weight.data.normal_(0, 0.01)
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1)
            module.bias.data.zero_()

    def forward(self, x):
        for m in self.hidden_blocks:
            x = m(x)

        out = []
        for m in self.out_blocks:
            x = m(x)
            if isinstance(m, DWBlock):
                out.append(x)

        return out

class TFPN(nn.Module):
    def __init__(
        self,
        tfpn_channels, # this should match hidden_channels in the Backbone
        num_outputs = 3,  # should match num_outputs in the Backbone
        device = 'cpu'
    ):
        super().__init__()
        self.tfpn_convs = nn.ModuleList(
            [
                DWUnit(tfpn_channels, tfpn_channels, device=device) for _ in range(num_outputs)
            ]
        )
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            if module.bias is not None:
                nn.init.xavier_normal_(module.weight.data)
                module.bias.data.fill_(0.02)
            else:
                module.weight.data.normal_(0, 0.01)
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1)
            module.bias.data.zero_()
    

    def forward(self, features):
        num_feature = len(features) # must be equal to num_outputs

        for i in range(num_feature - 1, 0, -1):
            features[i] = self.tfpn_convs[i](features[i])
            features[i - 1] = features[i - 1] + F.interpolate(features[i], size = features[i - 1].shape[2:4], mode = 'nearest-exact')

        features[0] = self.tfpn_convs[0](features[0])
        return features

class DecoupleHead(nn.Module):
    def __init__(
        self,
        in_channels, # matches tfpn_channels in TFPN
        num_levels = 3, # matches num_outputs in TFPN
        num_shared = 1, # 1 in the original publication
        num_stacked = 1,# 1 in the original publication
        device = 'cpu'
    ):
        super().__init__()
        self.shared_convs = nn.Sequential(
            OrderedDict(
                [
                    ( f'shconv{i}', DWUnit(in_channels, in_channels, device=device) )
                    for i in range(num_levels)
                ]
            )
        )

        in_ch = in_channels
        out_channels = [ in_channels for _ in range(num_stacked) ]

        out_channels[num_stacked - 1] = 1 # class prediction

        # We will flatten the spatial dimensions to compute loss by iterating over all anchor positions for all levels
        self.flatten = nn.Flatten(2, -1)

        self.cls_convs = nn.ModuleList(
            [
                nn.Sequential(
                    OrderedDict(
                        [
                            ( f'clsconv{i}_{j}', DWUnit(in_ch, out_ch, out_ch == in_ch, device=device) ) # out_ch == in_ch for hidden layers, != for the last layer
                            for j, out_ch in enumerate(out_channels)
                        ]
                    )
                )
                for i in range(num_levels)
            ]
        )

        out_channels[num_stacked - 1] = 4 # box prediction

        self.box_convs = nn.ModuleList(
            [
                nn.Sequential(
                    OrderedDict(
                        [
                            ( f'boxconv{i}_{j}', DWUnit(in_ch, out_ch, out_ch == in_ch, device=device) ) # out_ch == in_ch for hidden layers, != for the last layer
                            for j, out_ch in enumerate(out_channels)
                        ]
                    )
                )
                for i in range(num_levels)
            ]
        )
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            if module.bias is not None:
                nn.init.xavier_normal_(module.weight.data)
                module.bias.data.fill_(0.02)
            else:
                module.weight.data.normal_(0, 0.01)
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1)
            module.bias.data.zero_()
 

    def forward(self, features):
        num_feature = len(features) # must be equal to num_levels

        for i in range(num_feature):
            features[i] = self.shared_convs[i](features[i])

        cls_preds = []
        box_preds = []
        for i in range(num_feature):
            cls = self.cls_convs[i](features[i])

            cls = self.flatten(cls)
            cls_preds.append(torch.permute(cls, (0, 2, 1)))
            box = self.box_convs[i](features[i])
            box = self.flatten(box)
            box_preds.append(torch.permute(box, (0, 2, 1)))

        # Predictions have shape: (nbatch, sum WxH for all levels, 1)
        cls_flat = torch.cat(cls_preds, 1)

        # shape: (nbatch, sum WxH for all levels, 4)
        box_flat = torch.cat(box_preds, 1)

        return cls_flat, box_flat

class YunetModel(nn.Module):
    def __init__(self,
        backbone_layers:int = 1,
        tfpn_levels:int = 3,
        head_channels:int = 16,
        hidden_channels:int = 64,
        device = 'cpu'
    ):
        super().__init__()
        self.model = nn.Sequential(
            OrderedDict(
                [
                    ( 'head',  ConvHead(3, head_channels, head_channels, device=device) ),
                    ( 'bbone', Backbone(head_channels, hidden_channels, backbone_layers, tfpn_levels, device=device) ),
                    ( 'tfpn',  TFPN(hidden_channels, tfpn_levels, device=device) ),
                    ( 'dcoup', DecoupleHead(hidden_channels, tfpn_levels, device=device) )
                ]
            )
        )
    def forward(self, input_image: torch.Tensor) -> tuple:
        preds = self.model(input_image)
        return preds

