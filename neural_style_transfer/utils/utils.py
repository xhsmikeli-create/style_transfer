import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
import os
import matplotlib.pyplot as plt


from models.definitions.vgg_nets import Vgg19


IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]


# 加载图像、预处理、保存图像等实用函数。
def load_image(img_path, target_shape=None):
    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')
    # 用 np.fromfile + cv.imdecode 可以更稳地处理中文路径。
    img_bytes = np.fromfile(img_path, dtype=np.uint8)
    img = cv.imdecode(img_bytes, cv.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'Failed to load image: {img_path}')
    img = img[:, :, ::-1]  # [:, :, ::-1] converts BGR (opencv format...) into RGB

    if target_shape is not None:
        # 传入整数时，约定它代表目标高度，宽度按原图比例自动缩放。
        if isinstance(target_shape, int) and target_shape != -1:
            current_height, current_width = img.shape[:2]
            new_height = target_shape
            new_width = int(current_width * (new_height / current_height))
            img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        else:
            # 传入二元尺寸时，直接缩放到指定高宽。
            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)

    # 缩放后再做数值归一化。
    img = img.astype(np.float32)
    img /= 255.0
    return img


def prepare_img(img_path, target_shape, device):
    img = load_image(img_path, target_shape=target_shape)

    # 先转 tensor，再乘回 255，然后减去 ImageNet 均值。
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)
    ])

    img = transform(img).to(device).unsqueeze(0)

    return img


def save_image(img, img_path):
    if len(img.shape) == 2:
        # 单通道图像可视化时，复制成 3 通道保存。
        img = np.stack((img,) * 3, axis=-1)
    success, encoded = cv.imencode(os.path.splitext(img_path)[1], img[:, :, ::-1])
    if not success:
        raise ValueError(f'Failed to encode image for saving: {img_path}')
    encoded.tofile(img_path)


def generate_out_img_name(config, is_reconstruction=False):
    prefix = os.path.basename(config['content_img_name']).split('.')[0] + '_' + os.path.basename(config['style_img_name']).split('.')[0]
    if is_reconstruction:
        suffix = f'_o_adam_h_{str(config["height"])}_m_vgg19{config["img_format"][1]}'
    else:
        suffix = f'_o_adam_i_{config["init_method"]}_h_{str(config["height"])}_m_vgg19_cw_{config["content_weight"]}_sw_{config["style_weight"]}_tv_{config["tv_weight"]}{config["img_format"][1]}'
    return prefix + suffix


def save_and_maybe_display(optimizing_img, dump_path, config, img_id, num_of_iterations, should_display=False, is_reconstruction=False):
    saving_freq = config['saving_freq']
    out_img = optimizing_img.squeeze(axis=0).to('cpu').detach().numpy()
    out_img = np.moveaxis(out_img, 0, 2)

    # saving_freq=-1 表示只保存最后一张，否则按固定频率保存中间过程。
    if img_id == num_of_iterations-1 or (saving_freq > 0 and img_id % saving_freq == 0):
        img_format = config['img_format']
        out_img_name = str(img_id).zfill(img_format[0]) + img_format[1] if saving_freq != -1 else generate_out_img_name(config, is_reconstruction=is_reconstruction)
        dump_img = np.copy(out_img)
        # 把前面减掉的 ImageNet 均值再加回来，恢复成可显示的像素空间。
        dump_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
        dump_img = np.clip(dump_img, 0, 255).astype('uint8')
        out_path = os.path.join(dump_path, out_img_name)
        success, encoded = cv.imencode(os.path.splitext(out_path)[1], dump_img[:, :, ::-1])
        if not success:
            raise ValueError(f'Failed to encode image for saving: {out_path}')
        encoded.tofile(out_path)

    if should_display:
        plt.imshow(np.uint8(get_uint8_range(out_img)))
        plt.show()


def get_uint8_range(x):
    if isinstance(x, np.ndarray):
        # 把任意数值范围线性拉伸到 0~255，便于显示特征图或 Gram 矩阵。
        x -= np.min(x)
        x /= np.max(x)
        x *= 255
        return x
    else:
        raise ValueError(f'Expected numpy array got {type(x)}')



# 计算内容损失、风格损失、总变差损失的函数，以及准备模型和目标表示的函数。
def prepare_model(device):
    # 优化式 NST 不训练 VGG 权重，只把它当作固定的特征提取器。
    model = Vgg19(requires_grad=False, show_progress=True)

    content_feature_maps_index = model.content_feature_maps_index
    style_feature_maps_indices = model.style_feature_maps_indices
    layer_names = model.layer_names
    # 记录内容图和风格图的目标表示所在的层数和层名，后面优化时直接用这些索引访问对应的特征图。
    content_fms_index_name = (content_feature_maps_index, layer_names[content_feature_maps_index])
    style_fms_indices_names = (style_feature_maps_indices, layer_names)
    return model.to(device).eval(), content_fms_index_name, style_fms_indices_names


def gram_matrix(x, should_normalize=True):
    # 把空间维度展平后做特征相关性统计，得到风格表示。
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if should_normalize:
        gram /= ch * h * w
    return gram


def total_variation(y):
    # 计算相邻像素差异，惩罚过于尖锐的局部变化。
    return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + \
           torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))
