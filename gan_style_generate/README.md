# CycleGAN Art Style Transfer

This project keeps only the CycleGAN path needed for the homework:

- train an unpaired CycleGAN model
- run a trained generator on input photos
- generate Monet / Van Gogh / Ukiyo-e / comic-book style images

## Main Files

- `train.py`: trains a CycleGAN model.
- `generate.py`: loads a trained generator and saves generated images.
- `models/cycle_gan_model.py`: CycleGAN training logic.
- `models/test_model.py`: single-generator inference logic.
- `models/networks.py`: generator, discriminator, and GAN loss definitions.
- `data/unaligned_dataset.py`: reads unpaired training folders.
- `data/single_dataset.py`: reads one folder of input images for generation.
- `model_weights/`: stores pretrained and trained model weights.
- `datasets/`: stores training and input images.
- `results/`: stores generated images.

## Dataset Layout

CycleGAN training expects unpaired folders:

```text
datasets/monet2photo/
  trainA/  Monet paintings
  trainB/  photos
  testA/   Monet test images
  testB/   photo test images
```

For `monet2photo`, `vangogh2photo`, `ukiyoe2photo`, and `anime2photo`, domain `A` is the target style images and domain `B` is photos.
For `superhero2photo`, domain `A` is public-domain comic pages and domain `B` is photos.

## Train

For the anime dataset, put anime-style images in `datasets/anime2photo/trainA` and photos in `datasets/anime2photo/trainB`.
If you have raw folders, this command can split and copy them into the correct layout:

```powershell
python datasets/prepare_anime2photo.py --anime_raw D:\data\anime_raw --photo_raw D:\data\photo_raw
```

```powershell
conda activate pytorch-img2img
python train.py --dataroot datasets/monet2photo --name monet_cyclegan --model cycle_gan --batch_size 1
```

```powershell
python train.py --dataroot datasets/vangogh2photo --name vangogh_cyclegan --model cycle_gan --batch_size 1
```

```powershell
python train.py --dataroot datasets/ukiyoe2photo --name ukiyoe_cyclegan --model cycle_gan --batch_size 1
```

```powershell
python train.py --dataroot datasets/anime2photo --name anime_cyclegan --model cycle_gan --batch_size 1
```

```powershell
python train.py --dataroot datasets/superhero2photo --name superhero_cyclegan --model cycle_gan --batch_size 1
```

Training output is saved under:

```text
model_weights/<experiment_name>/
```

## Generate

For pretrained Monet style:

```powershell
python generate.py --dataroot datasets/monet2photo/testB --name style_monet_pretrained --model test --no_dropout --preprocess scale_width --load_size 512
```

For pretrained Van Gogh style:

```powershell
python generate.py --dataroot datasets/vangogh2photo/testB --name style_vangogh_pretrained --model test --no_dropout --preprocess scale_width --load_size 512
```

For a model trained on one of these datasets, photos are in domain `B`, so photo-to-painting uses `G_B`:

```powershell
python generate.py --dataroot datasets/monet2photo/testB --name monet_cyclegan --model test --model_suffix _B --no_dropout --preprocess scale_width --load_size 512
```

```powershell
python generate.py --dataroot datasets/ukiyoe2photo/testB --name ukiyoe_cyclegan --model test --model_suffix _B --no_dropout --preprocess scale_width --load_size 512
```

```powershell
python generate.py --dataroot datasets/anime2photo/testB --name anime_cyclegan --model test --model_suffix _B --no_dropout --preprocess scale_width --load_size 512
```

```powershell
python generate.py --dataroot datasets/superhero2photo/testB --name superhero_cyclegan --model test --model_suffix _B --no_dropout --preprocess scale_width --load_size 512
```

Generated images are saved under:

```text
results/<experiment_name>/test_latest/images/
```
