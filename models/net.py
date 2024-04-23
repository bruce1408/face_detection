import time
import torch
import math
import torch.nn as nn
from typing import List
from torch.nn.functional import gelu, layer_norm
from einops.layers.torch import Rearrange, Reduce
import torchvision.models._utils as _utils
import torchvision.models as models
import torch.nn.functional as F
from torch import nn, Tensor
from torch.autograd import Variable

class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 16,
        emb_size: int = 768,
        img_size: int = 640,
    ):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )  # this breaks down the image in s1xs2 patches, and then flat them
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        # self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        # cls_tokens = repeat(self.cls_token, "() n e -> b n e", b=b)
        # x = torch.cat([cls_tokens, x], dim=1)  # prepending the cls token
        x += self.positions
        return x


class MLP(nn.Module):
    def __init__(self, input_dims, hidden_units, dropout_rate):
        super().__init__()
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.layers = nn.ModuleList()
        previous_dim = input_dims
        for units in self.hidden_units:
            self.layers.append(nn.Linear(in_features=previous_dim, out_features=units))
            self.layers.append(nn.Dropout(p=self.dropout_rate))
            previous_dim = units

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ViTforObjectDetection(nn.Module):
    def __init__(
        self,
        mlp_head_units: List[int],
        patch_size: int = 16,
        emb_size: int = 768,
        img_size: int = 224,
        num_heads: int = 4,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.inputs = torch.Tensor()
        self.layer_norm = nn.LayerNorm(emb_size)
        
        self.image_size = img_size
        self.patch_size = patch_size
        self.emb_size = emb_size
        
        self.to_patch_embedding = PatchEmbedding(
            patch_size=patch_size, img_size=img_size, emb_size=emb_size
        )
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=emb_size,
                nhead=num_heads,
                activation=gelu,
                layer_norm_eps=1e-6,
                norm_first=True,
            ),
            num_layers=depth,
        )
        
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.3)
        
        num_patches = math.ceil(img_size / patch_size) * math.ceil(img_size / patch_size)
        flattened_dim = num_patches * emb_size
        
        # self.mlp = MLP(mlp_head_units, dropout_rate=0.3)
        self.mlp = MLP(flattened_dim, mlp_head_units, dropout_rate=0.3)
        
        # 转置卷积层来上采样到80x80
        self.up_to_80 = nn.ConvTranspose2d(emb_size, 64, kernel_size=3, stride=2, padding=1, output_padding=1)

        # 1x1卷积用于通道数变换，从768到128，因为输出尺寸40x40是直接可用的
        self.conv_to_128 = nn.Conv2d(emb_size, 128, kernel_size=1)

        # 最大池化层来下采样到20x20
        self.conv_to_256 = nn.Conv2d(emb_size, 256, kernel_size=1)
        self.down_to_20 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, img)->Tensor:
        x = self.to_patch_embedding(img)
        x = self.transformer(x)
        x = self.layer_norm(x)
        x = self.flatten(x)
        x = self.dropout(x)
        
        
        # 输出多尺寸特征
        # print(x.shape, math.ceil(self.image_size / self.patch_size))
        x = x.reshape(-1, self.emb_size, math.ceil(self.image_size / self.patch_size), math.ceil(self.image_size / self.patch_size))
        
        # 生成80x80输出
        out1 = self.up_to_80(x)
        out1 = F.relu(out1)  # 激活函数

        # 生成40x40输出
        out2 = self.conv_to_128(x)
        out2 = F.relu(out2)  # 激活函数

        # 生成20x20输出
        out3 = self.conv_to_256(x)
        out3 = self.down_to_20(out3)
        out3 = F.relu(out3)  # 激活函数
        
        return out1, out2, out3
    
def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
    )

class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1, leaky = leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out

class FPN(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FPN,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride = 1, leaky = leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride = 1, leaky = leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride = 1, leaky = leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky = leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky = leaky)

    def forward(self, input):
        # names = list(input.keys())
        if isinstance(input, dict):
            input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out



class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, 2, leaky = 0.1),    # 3
            conv_dw(8, 16, 1),   # 7
            conv_dw(16, 32, 2),  # 11
            conv_dw(32, 32, 1),  # 19
            conv_dw(32, 64, 2),  # 27
            conv_dw(64, 64, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1), # 59 + 32 = 91
            conv_dw(128, 128, 1), # 91 + 32 = 123
            conv_dw(128, 128, 1), # 123 + 32 = 155
            conv_dw(128, 128, 1), # 155 + 32 = 187
            conv_dw(128, 128, 1), # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2), # 219 +3 2 = 241
            conv_dw(256, 256, 1), # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x

