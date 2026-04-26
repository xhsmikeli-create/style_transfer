# 项目快速部署

该项目包含两种神经风格迁移方法：

- `neural_style_transfer/`：经典优化式神经风格迁移，直接优化输出图像，效果可控但优化速度较慢。
- `fast_neural_style_transfer/`：快速风格迁移，使用已训练好的模型快速生成结果。

## 1. 创建环境

使用Anaconda创建虚拟环境，并安装依赖库：

```bash
conda create -n style-transfer python=3.10 -y
conda activate style-transfer
pip install -r requirements.txt
```

## 2. 启动 Web 前端

```bash
python web_app\app.py
```

使用浏览器打开：

```text
http://127.0.0.1:7860
```

页面左侧是经典优化式风格迁移，右侧是快速前馈式风格迁移。生成结果会保存到各自项目的输出目录。

## 3. 运行经典风格迁移

```bash
cd .\neural_style_transfer
python neural_style_transfer.py --content_img_name bear.jpg --style_img_name candy.jpg --max_iterations_adam 300 --saving_freq -1
```

输出图片会保存以下文件夹中：

```text
neural_style_transfer/data/output-images/
```

常用参数：

- `--content_img_name`：内容图文件名，放在 `neural_style_transfer/data/content-images/`
- `--style_img_name`：风格图文件名，放在 `neural_style_transfer/data/style-images/`
- `--height`：输出高度，越大越图像越清晰，但是耗时更长
- `--max_iterations_adam`：迭代次数，越大效果越充分
- `--saving_freq -1`：只保存最终结果，不输出中间过程

## 4. 运行快速风格迁移

快速风格迁移分为两个阶段，第一阶段为模型训练阶段，第二阶段为推理阶段。
* 其中训练阶段使用的数据集需要是 `ImageFolder` 结构，例如：

```text
D:\dataset\val2017\
  train\
    0001.jpg
    0002.jpg
    0003.jpg
```

训练命令：

```bash
python neural_style\neural_style.py train --dataset D:\dataset\my_train_data --style-image images\style-images\candy.jpg --save-model-dir saved_models --checkpoint-model-dir checkpoints --epochs 2 --batch-size 4 --image-size 256 --style-size 256 --content-weight 1e5 --style-weight 1e10 --lr 1e-3 --accel
```

训练完成后，模型会保存到：

```text
fast_neural_style_transfer/saved_models/
```

然后可以把生成的 `.model` 文件用于推理
```bash
python neural_style\neural_style.py eval --content-image images\content-images\amber.jpg --model saved_models\candy.pth --output-image output\amber-candy.jpg --accel
```

输出图片会保存到：

```text
fast_neural_style_transfer/output/
```

更多推理示例：

```bash
python neural_style\neural_style.py eval --content-image images\content-images\bear.jpg --output-image output\bear_my_style.jpg --model saved_models\good.model --accel
```

```bash
python neural_style\neural_style.py eval --content-image images\content-images\bear.jpg --output-image output\bear_candy.jpg --model saved_models\candy.pth --accel
```

```bash
python neural_style\neural_style.py eval --content-image images\content-images\golden_gate.jpg --output-image output\golden_gate_candy.jpg --model saved_models\candy.pth --accel
```

## 5. 加入自己的图片

经典风格迁移：

- 内容图放到 `neural_style_transfer/data/content-images/`
- 风格图放到 `neural_style_transfer/data/style-images/`

快速风格迁移：

- 内容图放到 `fast_neural_style_transfer/images/content-images/`
- 直接使用 `saved_models/` 里的 `.pth` 或者训练好的`.model`模型

## 6. 参考仓库

本项目学习和改写时主要参考了以下开源仓库：

- 经典优化式神经风格迁移：<https://github.com/gordicaleksa/pytorch-neural-style-transfer>
- 快速神经风格迁移：<https://github.com/pytorch/examples/tree/main/fast_neural_style>
