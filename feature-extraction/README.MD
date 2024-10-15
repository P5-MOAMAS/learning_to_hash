# Credits
[Original repository](https://github.com/Horizon2333/imagenet-autoencoder?tab=readme-ov-file#install)

# Setup
In "_/imagenet-autoencoder_" run to automatically setup project:
```bash
pip install -e .
```

## Downloading the pre-trained model
[Download link](https://drive.google.com/file/d/1WwJiQ1kBcNCZ37F6PJ_0bIL0ZeU3_sV8/view). Put the pth file in "/feature-extraction/imagenet-autoencoder/**_results_**"

# Running the encoder
```bash
python imagenet-autoencoder/tools/encode.py --arch vgg16 --resume imagenet-autoencoder/results/imagenet-vgg16.pth --img_path imagenet-autoencoder/figs/reconstruction.jpg
```
