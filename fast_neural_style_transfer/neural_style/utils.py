import torch
from PIL import Image

try:
    # Pillow 新版本使用 Image.Resampling.LANCZOS。
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    # 兼容旧版本 Pillow。
    RESAMPLE_LANCZOS = Image.ANTIALIAS

# 图片加载、保存和预处理的工具函数，以及 Gram 矩阵计算和 TV 损失计算。
def load_image(filename, size=None, scale=None):

    img = Image.open(filename).convert('RGB')
    if size is not None:
        img = img.resize((size, size), RESAMPLE_LANCZOS)
    elif scale is not None:
        # 推理时可以按比例缩小内容图，减少显存占用。
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), RESAMPLE_LANCZOS)
    return img


def save_image(filename, data):
    # 保存图片。
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(y):
    # 计算 Gram 矩阵。
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    # VGG 模型要求输入图像前先做归一化，减去 ImageNet 数据集的均值，除以标准差。
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std
