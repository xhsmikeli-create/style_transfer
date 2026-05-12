from collections import namedtuple
import torch
from torchvision.models import VGG19_Weights, vgg19

"""
    这里对 VGG19 做了一个“按层切片”的封装，方便直接拿到论文需要的中间层特征。
    如果你想对照 PyTorch 官方实现里的层编号，可以看：

    https://github.com/pytorch/vision/blob/3c254fb7af5f8af252c24e89949c54a3461ff0be/torchvision/models/vgg.py
"""


class Vgg19(torch.nn.Module):
    """
    这个类只暴露经典 NST 真正会用到的几层。

    风格表示使用多层特征，内容表示使用单个内容层。
    """
    def __init__(self, requires_grad=False, show_progress=False, use_relu=True):
        super().__init__()
        # 直接加载 torchvision 预训练的 VGG19 特征提取部分。
        vgg_pretrained_features = vgg19(weights=VGG19_Weights.DEFAULT, progress=show_progress).features
        if use_relu:
            # 用 relu 命名更容易和很多教程里的层名对应起来。
            self.layer_names = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'conv4_2', 'relu5_1']
            self.offset = 1
        else:
            self.layer_names = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv4_2', 'conv5_1']
            self.offset = 0
        # 内容层固定使用 conv4_2。
        self.content_feature_maps_index = 4
        # 风格层使用除 conv4_2 之外的所有暴露层。
        self.style_feature_maps_indices = list(range(len(self.layer_names)))
        self.style_feature_maps_indices.remove(4)

        # 按照论文需要的层位置，把完整 VGG 切成 6 段顺序模块。
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        for x in range(1+self.offset):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(1+self.offset, 6+self.offset):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(6+self.offset, 11+self.offset):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(11+self.offset, 20+self.offset):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(20+self.offset, 22):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        for x in range(22, 29++self.offset):
            self.slice6.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            # 这里只做特征提取，不更新参数。
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        # 顺序前向传播，并把每个关心的中间层输出都保留下来。
        x = self.slice1(x)
        layer1_1 = x
        x = self.slice2(x)
        layer2_1 = x
        x = self.slice3(x)
        layer3_1 = x
        x = self.slice4(x)
        layer4_1 = x
        x = self.slice5(x)
        conv4_2 = x
        x = self.slice6(x)
        layer5_1 = x
        # 用 namedtuple 返回多层结果，调用端可以像属性一样访问各层特征。
        vgg_outputs = namedtuple("VggOutputs", self.layer_names)
        out = vgg_outputs(layer1_1, layer2_1, layer3_1, layer4_1, conv4_2, layer5_1)
        return out
