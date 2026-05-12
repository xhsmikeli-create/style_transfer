# anime2photo Dataset

This folder follows the same CycleGAN layout as `monet2photo` and `vangogh2photo`.

```text
datasets/anime2photo/
  trainA/  anime-style images used as the target style domain
  trainB/  real photos used as the source photo domain
  testA/   held-out anime-style images
  testB/   held-out real photos for photo-to-anime generation
```

For training photo-to-anime style transfer:

- Domain `A` = anime images.
- Domain `B` = real photos.
- Put most images in `trainA` and `trainB`.
- Put a small held-out sample in `testA` and `testB`.

Current local install:

- Anime source: `huggan/anime-faces` from Hugging Face, license `cc0-1.0`.
- Photo source: local `datasets/monet2photo/trainB`.
- `trainA`: 1440 anime images.
- `trainB`: 1440 photo images.
- `testA`: 160 anime images.
- `testB`: 160 photo images.
- Download cache: `datasets/_downloads/anime-faces/data.zip`.

If you already have two raw folders, one for anime images and one for photos, this helper can split and copy them into the CycleGAN layout:

```powershell
python datasets/prepare_anime2photo.py --anime_raw D:\data\anime_raw --photo_raw D:\data\photo_raw
```

Train:

```powershell
python train.py --dataroot datasets/anime2photo --name anime_cyclegan --model cycle_gan --batch_size 1
```

Generate photo-to-anime images after training:

```powershell
python generate.py --dataroot datasets/anime2photo/testB --name anime_cyclegan --model test --model_suffix _B --no_dropout --preprocess scale_width --load_size 512
```

Generated images will be saved under:

```text
results/anime_cyclegan/test_latest/images/
```

