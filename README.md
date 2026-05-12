# 神经风格迁移项目

本项目包含三种风格迁移/图像生成方法，并提供一个统一的 Web 前端：

- `neural_style_transfer/`：经典优化式神经风格迁移，直接优化输出图像，效果可控但速度较慢。
- `fast_neural_style_transfer/`：快速前馈式风格迁移，使用已训练模型快速生成结果。
- `gan_style_generate/`：基于 CycleGAN/pix2pix 结构的 GAN 图像到图像推理。
- `web_app/`：浏览器前端，用同一套输入/输出目录管理图片。

## 1. 创建环境

建议使用单独的 Conda 环境，避免系统或 `base` 环境里的 Flask/Jinja2 版本冲突。

```bash
conda create -n style-transfer python=3.10 -y
conda activate style-transfer
pip install -r requirements.txt
```

## 2. 前端输入输出目录

Web 前端统一使用以下目录：

```text
data_box/
  input/    # 前端上传的输入图片
  output/   # 前端生成的输出图片
```

页面里的 `Upload` 会把图片保存到 `data_box/input/`。上传时会做基础清理和校验：

- 只允许 `.jpg`、`.jpeg`、`.png`
- 自动清理文件名，避免路径和特殊字符污染
- 限制单个上传文件最大 32 MB

页面里的 `Clean Input` 只清理 `data_box/input/` 里的图片，不会删除输出结果。`data_box/input/*` 和 `data_box/output/*` 已加入 `.gitignore`，运行时图片不会被误提交到 GitHub。

## 3. 启动 Web 前端

```bash
python web_app\app.py
```

浏览器打开：

```text
http://127.0.0.1:7860
```

如果 `7860` 被占用，可以换端口：

```bash
python web_app\app.py --port 7861
```

## 4. 使用 Web 前端

1. 点击 `Upload` 上传内容图和风格图。
2. `Classic Optimization` 选择内容图、风格图和参数后运行经典方法。
3. `Fast Transfer` 选择内容图和 `.pth/.model` 模型后运行快速方法。
4. `GAN Transfer` 选择内容图和 GAN 权重目录后运行 `pytorch-img2img` 推理。
5. 生成结果会显示在页面中，同时保存到 `data_box/output/`。

快速方法使用的模型放在：

```text
fast_neural_style_transfer/saved_models/
```

GAN 方法使用的权重目录放在：

```text
gan_style_generate/model_weights/
```

Web 后端会优先使用 `pytorch-img2img` Conda 环境的 Python 运行模型推理。如果你的环境路径不同，可以在启动前设置：

```bash
set PYTORCH_IMG2IMG_PYTHON=D:\path\to\envs\pytorch-img2img\python.exe
python web_app\app.py
```

## 5. 命令行运行经典方法

```bash
cd .\neural_style_transfer
python neural_style_transfer.py --content_img_name bear.jpg --style_img_name candy.jpg --max_iterations_adam 300 --saving_freq -1
```

默认输入输出目录：

```text
neural_style_transfer/data/content-images/
neural_style_transfer/data/style-images/
neural_style_transfer/data/output-images/
```

常用参数：

- `--content_img_name`：内容图文件名
- `--style_img_name`：风格图文件名
- `--height`：输出高度，越大越清晰但越慢
- `--max_iterations_adam`：迭代次数
- `--saving_freq -1`：只保存最终结果

## 6. 命令行运行快速方法

推理示例：

```bash
cd .\fast_neural_style_transfer
python neural_style\neural_style.py eval --content-image images\content-images\bear.jpg --output-image output\bear_good.jpg --model saved_models\good.model --accel
```

训练数据需要是 `ImageFolder` 结构，例如：

```text
D:\dataset\val2017\
  train\
    0001.jpg
    0002.jpg
```

训练示例：

```bash
python neural_style\neural_style.py train --dataset D:\dataset\my_train_data --style-image images\style-images\candy.jpg --save-model-dir saved_models --checkpoint-model-dir checkpoints --epochs 2 --batch-size 4 --image-size 256 --style-size 256 --content-weight 1e5 --style-weight 1e10 --lr 1e-3 --accel
```

## 7. 命令行运行 GAN 推理

```bash
cd .\gan_style_generate
python generate.py --dataroot ..\data_box\input --name style_vangogh_pretrained --checkpoints_dir model_weights --results_dir ..\data_box\output\gan_runs --model test --dataset_mode single --num_test 1 --preprocess none --no_dropout --eval
```

Web 前端会自动把选中的单张输入图复制到临时目录，所以不会误处理 `data_box/input/` 里的其它图片。前端最终展示的是 GAN 输出中的 `*_fake.png`。

## 8. 参考仓库

- 经典优化式神经风格迁移：<https://github.com/gordicaleksa/pytorch-neural-style-transfer>
- 快速神经风格迁移：<https://github.com/pytorch/examples/tree/main/fast_neural_style>
- GAN 图像到图像迁移：<https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix>
