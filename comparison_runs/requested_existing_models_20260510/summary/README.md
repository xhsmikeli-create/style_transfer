# Requested Comparison, Existing Models Only

No model training was performed for this run.

## Inputs

- `mountain.jpg`: copied from `gan_style_generate/datasets/monet2photo/testB/2014-08-17 23_34_43.jpg`
- `golden_gate.jpg`: copied from `fast_neural_style_transfer/images/content-images/golden_gate.jpg`
- `lion.jpg`: copied from `neural_style_transfer/data/content-images/lion.jpg`

## Models Used

Fast:

- Ukiyoe: `fast_neural_style_transfer/saved_models/ukiyoe.model`

GAN:

- Van Gogh: `gan_style_generate/model_weights/style_vangogh_pretrained/latest_net_G.pth`
- Ukiyoe: `gan_style_generate/model_weights/style_ukiyoe_pretrained/latest_net_G.pth`
- Anime: `gan_style_generate/model_weights/style_anime_pretrained/latest_net_G.pth`

Note: Fast Van Gogh was not generated because no existing Fast Van Gogh model was found under `fast_neural_style_transfer/saved_models`, and this run follows the no-training constraint.

## Summary Figures

- `full_comparison_existing_models.jpg`
- `ukiyoe_fast_vs_gan_existing_models.jpg`
- `gan_styles_existing_models.jpg`
- `selected_inputs_existing_models.jpg`

