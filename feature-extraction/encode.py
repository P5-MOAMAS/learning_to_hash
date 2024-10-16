import gc
import os
import sys
import time

import argparse
import torch
from torch import nn
import vgg
from torchvision.transforms import transforms
from PIL import Image

from dataset import get_dataset_batch_amount, DynamicDataset

# https://stackoverflow.com/questions/3160699/python-progress-bar
def progressbar(it, prefix="", size=60, out=sys.stdout):
    count = len(it)
    start = time.time() # time estimate start
    def show(j):
        x = int(size*j/count)
        # time estimate calculation and string
        remaining = ((time.time() - start) / j) * (count - j)
        mins, sec = divmod(remaining, 60) # limited to minutes
        time_str = f"{int(mins):02}:{sec:03.1f}"
        print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count} Est wait {time_str}", end='\r', file=out, flush=True)
    show(0.1) # avoid div/0
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)


def load_dict(path: str, model: nn.Module, device: str):
    if os.path.isfile(path):
        if device.type == 'cuda' :
            checkpoint = torch.load(path, weights_only=False)
        else:
            checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
        model_dict = model.state_dict()
        model_dict.update(checkpoint['state_dict'])
        model.load_state_dict(model_dict)
        del checkpoint
    else:
        sys.exit("No model found on path: " + path)
    return model


def encode(model: nn.Module, img):
    with torch.no_grad():
        code = model.module.encoder(img).cpu().numpy()
    return code


def get_model(device: str):
    model = vgg.VGGAutoEncoder(vgg.get_configs())
    model = nn.DataParallel(model).to(device)

    return model


# @TODO Ensure multithreading
def encode_dataset(device: str, dataset_name: str):
    model = load_model(device)
    encoded_images = []
    batches = get_dataset_batch_amount(dataset_name);

    for batch in range(1, batches + 1):
        print("Loading batch " + str(batch) + "/" + str(batches) + " of " + dataset_name)
        data = DynamicDataset(dataset_name, batch_number=batch)
        print("Starting encoding for " + str(len(data)) + " images")

        for img in progressbar(data):
            img = img.to(device)
            code = encode(model, img)
            encoded_images.append(code)

        # Delete the dataset as soon as the images have been encoded! Saves memory.
        del data
        gc.collect()

        print("Finished encoding for batch " + str(batch) + " of " + dataset_name)

    return encoded_images


def load_model(device):
    model = get_model(device)
    model.eval()
    load_dict("feature-extraction/imagenet-vgg16.pth", model, device)
    return model


def encode_image(image_path: str, device: str):
    model = load_model(device)

    # Transform image to match model input size
    trans = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()
                  ])
    img = Image.open(image_path).convert("RGB")
    img = trans(img).unsqueeze(0).to(device)
    code = encode(model, img)

    return code


def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Based Image Encoder')
    parser.add_argument("-ip", "--image-path", type=str, help="path to image")
    parser.add_argument("-d", "--dataset", type=str, help="dataset to use (cifar-10)", default="cifar-10")
    parser.add_argument("-fc", "--force-cpu", action="store_true", help="force cpu")
    parser.add_argument("-fg", "--force-gpu", action="store_true", help="force gpu")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.force_cpu:
        device = torch.device("cpu")
    elif args.force_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            sys.exit("No GPU found")

    result = None
    if args.image_path:
        result = encode_image(args.image_path, device)
        print("Images encoded: " + str(len(result)))
    elif args.dataset:
        result = encode_dataset(device, args.dataset)

    return result


if __name__ == '__main__':
    main()
