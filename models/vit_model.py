import torch
import warnings
import math
warnings.filterwarnings("ignore")
# from vit_pytorch import ViT
from typing import List
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import nn, Tensor
from torch.nn.functional import gelu, layer_norm
from torch.nn.modules import Transformer, TransformerEncoder
from torch.nn.modules.transformer import F


import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict

from models.net import MobileNetV1 as MobileNetV1
from models.net import FPN as FPN
from models.net import SSH as SSH


# from net import MobileNetV1 as MobileNetV1
# from net import FPN as FPN
# from net import SSH as SSH

class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512, num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors*4, kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 10)

class RetinaFace(nn.Module):
    def __init__(self, cfg = None, phase = 'train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace,self).__init__()
        self.phase = phase
        
        backbone = ViTforObjectDetection(
            mlp_head_units=[768, 256, 64],
            patch_size=cfg["patch_size"],
            emb_size=cfg["emb_size"],
            img_size=cfg["image_size"],
            num_heads=cfg["num_heads"],
            depth=4
        )
        self.image_size = cfg["image_size"]
        self.emb_size = cfg["emb_size"]
        self.patch_size = cfg["patch_size"]
        self.body = backbone
        
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]

        out_channels = cfg['out_channel']

        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)
        
        fpn_num = 3
        self.ClassHead = self._make_class_head(fpn_num, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num, inchannels=cfg['out_channel'])


    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3, inchannels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3, inchannels=64, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead
    
    def forward(self, inputs):
        out = self.body(inputs) # 512, 105, 105 -- 1024, 53, 53 --- 2048,27,27
                
        # # FPN
        fpn = self.fpn(out)        
        
        # # SSH
        feature1 = self.ssh1(fpn[0])# ([1, 256, 105, 105])
        feature2 = self.ssh2(fpn[1])# ([1, 256, 53, 53])
        feature3 = self.ssh3(fpn[2])# ([1, 256, 27, 27])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)
        
        # print(out[0].shape)  # torch.Size([1, 512, 105, 105])
        # print(out[1].shape)  # torch.Size([1, 1024, 53, 53])
        # print(out[2].shape)  # torch.Size([1, 2048, 27, 27])
        # print("fpn 1 shape is ",fpn[0].shape) # fpn 1 shape is  torch.Size([1, 256, 105, 105])
        # print("fpn 2 shape is ",fpn[1].shape) # fpn 2 shape is  torch.Size([1, 256, 53, 53])
        # print("fpn 3 shape is ",fpn[2].shape) # fpn 3 shape is  torch.Size([1, 256, 27, 27])
        # print(bbox_regressions.shape)
        # print(classifications.shape)
        # print(ldm_regressions.shape)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output
        
        
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
        

    # 测试 ViTforObjectDetection
def test_vit_object_detection():
    # 配置
    img_size = 640
    batch_size = 1  # 测试 2 个图像
    img = torch.randn(batch_size, 3, img_size, img_size)  # 创建一个随机图像批次

    # 实例化模型
    model = ViTforObjectDetection(
        mlp_head_units=[768, 256, 64],
        patch_size=16,
        emb_size=768,
        img_size=img_size,
        num_heads=4,
        depth=4
    )

    # 将模型设置为评估模式
    model.eval()

    # 前向传播
    with torch.no_grad():
        outputs = model(img)
        
    
if __name__ == "__main__":


    # 执行测试
    test_vit_object_detection()
    
    # from torchvision.transforms import Compose, ToTensor, Normalize, Resize

    cfg = {
        'name': 'vit',  # 可以选择 'mobilenet0.25' 或 'Resnet50'
        'pretrain': False,  # 设置为True如果有预训练权重
        # 'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
        "patch_size":16,
        "emb_size":768,
        "image_size":640,
        'in_channel': 32,
        'out_channel': 64
    }
    
    model = RetinaFace(cfg, phase='test')
 
    # 创建一个随机的图像张量作为输入
    input_tensor = torch.rand(1, 3, 640, 640)  # Batch size 1
    # input_tensor = transform(input_tensor)  # 应用预处理

    # 设置模型为评估模式
    model.eval()

    # 执行前向传播，获取输出
    with torch.no_grad():
        outputs = model(input_tensor)  # 增加批次维度
        print(outputs[0].shape)
        print(outputs[1].shape)
        print(outputs[2].shape)