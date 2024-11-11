# Credits
[Original repository](https://github.com/Horizon2333/imagenet-autoencoder?tab=readme-ov-file#install)

# Setup
## Dependencies
Run to automatically setup project:
```bash
pip install -e .
```
### Using virtual environment
```bash
# Create virtual environment
python -m venv venv
# Activate environment
. venv/bin/activate
# Install dependencies
pip install -e .
```

## Downloading the pre-trained model
[Download link](https://drive.google.com/file/d/1WwJiQ1kBcNCZ37F6PJ_0bIL0ZeU3_sV8/view)
# Running the encoder
See all flags using:
```bash
python feature-extraction/encode.py -h
```

## Single image
```bash
python encode.py <image_path>

# Example
python feature-extraction/encode.py --image-path feature-extraction/reconstruction.jpg
```

## For datasets
```bash
python feature-extraction/encode.py --dataset cifar-10
```

