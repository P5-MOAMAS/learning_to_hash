import argparse
import os
import time
import sys
from typing import List

from torchvision import models, transforms
from torchvision.models import AlexNet_Weights
import torch.nn as nn
import torch
from triton.language import tensor
from torchvision import datasets
from torchvision.transforms import ToTensor, functional


class AlexNet(nn.Module):
    def __init__(self, feature_size):
        super(AlexNet, self).__init__()

        model_alexnet = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        self.features = model_alexnet.features
        cl1 = nn.Linear(256 * 6 * 6, 4096)
        cl1.weight = model_alexnet.classifier[1].weight
        cl1.bias = model_alexnet.classifier[1].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = model_alexnet.classifier[4].weight
        cl2.bias = model_alexnet.classifier[4].bias

        self.feature_layer = nn.Sequential(
            nn.Dropout(),
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Linear(4096, feature_size),
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.feature_layer(x)
        return x


    def query(self, image):
        self.eval()
        with torch.no_grad():
            image = image.to(torch.device("cuda:0"))
            output = self(image)
        return output


class Encoder:
    def __init__(self, args):
        self.dataset_name = args.dataset
        self.alex_net = AlexNet(512)
        self.start_time = 0


    @staticmethod
    def transform_image(image):
        trans = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return trans(image).unsqueeze(0)


    def print_elapsed_time(self):
        minutes, seconds = divmod(time.time() - self.start_time, 60)
        hours, minutes = divmod(minutes, 60)
        elapsed_time = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

        print("Total time elapsed: " + elapsed_time)


    def save_to_file(self, codes: List[tensor], dataset_batch: int):
        file = "features/" + str(self.dataset_name) + "-" + str(dataset_batch) + "-features"
        print("Saving features to " + file)
        torch.save(codes, file)
        print("Features successfully saved!\n")


    def encode(self):
        data = load_data_set(self.dataset_name)
        self.start_time = time.time()
        os.makedirs("features", exist_ok=True)

        codes = []
        for index, image in enumerate(data, start=1):
            image = self.transform_image(image)
            code = self.alex_net(image)
            # Detach ensures the code doesn't linger in memory
            code = code.detach().flatten()
            codes.append(code)
            print("Encoding image", index, "of", len(data), flush=True, end="\r")
            del code, image

        print()

        self.print_elapsed_time()
        self.save_to_file(codes, 1)

        del codes


def load_data_set(dataset_name: str) -> List:
    match dataset_name:
        case "cifar-10":
            data = datasets.CIFAR10(root="data", train=True, download=True).data
        case "mnist":
            data = datasets.FashionMNIST( root="data", train=True, download=True, transform=ToTensor()).data.numpy()
        case _:
            print("Unrecognized data set!")
            sys.exit(1)

    image_data = []
    for image in data:
        image_data.append(functional.to_pil_image(image).convert("RGB"))

    return image_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Based Image Encoder')
    parser.add_argument("-d", "--dataset", type=str, help="dataset to use (cifar-10), sets: cifar-10, mnist, image-net", default="cifar-10")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise Exception("No GPU found")

    if args.dataset:
        encoder = Encoder(args)
        encoder.encode()
    else:
        print("Specify a dataset using flag -d")
