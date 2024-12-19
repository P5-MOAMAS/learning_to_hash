import argparse
import math
import os
import sys
import time
from typing import List, Any

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import models, transforms
from torchvision.models import AlexNet_Weights
from torchvision.transforms import ToTensor, functional
from tqdm import tqdm, trange
from triton.language import tensor

from nuswide_loader import NuswideMLoader


class AlexNet(nn.Module):
    def __init__(self, feature_size, device=torch.device("cuda:0")):
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
        self.to(device)
        self.device = device

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.feature_layer(x)
        return x

    def query(self, image):
        self.eval()
        with torch.no_grad():
            image = image.to(self.device)
            output = self(image)
        return output


class Encoder:
    def __init__(self, args, device=torch.device("cuda:0")):
        self.dataset_name = args.dataset
        self.alex_net = AlexNet(512, device)
        self.start_time = 0

    """
    Transforms an image into the correct size and into a tensor
    """
    @staticmethod
    def transform_image(image):
        trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return trans(image).unsqueeze(0)

    """
    Prints the current elapsed time
    """
    def print_elapsed_time(self):
        minutes, seconds = divmod(time.time() - self.start_time, 60)
        hours, minutes = divmod(minutes, 60)
        elapsed_time = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

        print("Total time elapsed: " + elapsed_time)

    """
    Saves the features to file
    """
    def save_to_file(self, codes: List[tensor], dataset_batch: int):
        os.makedirs("features", exist_ok=True)
        file = "features/" + str(self.dataset_name) + "-" + str(dataset_batch) + "-features"
        print("Saving features to " + file)
        torch.save(codes, file)
        print("Features successfully saved!\n")


    """
    Encodes the entire dataset into feature vectors
    """
    def encode(self):
        data = load_data_set(self.dataset_name)
        self.start_time = time.time()

        batch_size = 1000
        codes = []

        # Go through all images in batches
        t = tqdm(range(0, len(data)), desc="Encoding images")
        for i in range(0, len(data), batch_size):
            batch_end = i + batch_size
            if batch_end >= len(data):
                batch_end = len(data)

            # Transform all images to the correct format for the entire batch
            batch = []
            for index in range(i, batch_end):
                batch.append(self.transform_image(data[index]))
                t.update(1)

            # Turn into tensor of tensors and run it through AlexNet
            code = self.alex_net(torch.stack(batch).squeeze(1).to(device))

            # Detach ensures the code doesn't linger in memory
            code = code.detach()
            codes.extend(code)
            del code
        t.close()
        self.print_elapsed_time()
        self.save_to_file(codes, 1)

        del codes

    """
    Calculate extraction time for a 1000 queries to get an average per image
    """
    def calculate_extraction_time(self):
        data = load_data_set(self.dataset_name)

        encoding_times = []
        processing_times = []
        for i in trange(1000, desc="Calculating extraction time"):
            # Calculate the time to transform images
            processing_time_start = time.time_ns()
            picture = self.transform_image(data[i])
            picture = picture.to(self.alex_net.device)
            processing_times.append(time.time_ns() - processing_time_start)

            # Calculate the encoding time for an image
            encoding_time_start = time.time_ns()
            self.alex_net(picture)
            encoding_times.append(time.time_ns() - encoding_time_start)

        # Calculate the mean encoding time and processing time
        mean_time = np.mean(encoding_times) * math.pow(10, -6)
        mean_processing_time = np.mean(processing_times) * math.pow(10, -6)

        print(f"{self.dataset_name} encoding times:"
              f"\n\tProcessing: {mean_processing_time:.3f} ms"
              f"\n\t{self.alex_net.device}: {mean_time:.3f} ms, total: {mean_processing_time + mean_time:.3f} ms")
        return mean_time


def load_data_set(dataset_name: str) -> NuswideMLoader | List[Any]:
    match dataset_name:
        case "cifar-10":
            data = datasets.CIFAR10(root="data", train=True, download=True).data
        case "mnist":
            data = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor()).data.numpy()
        case "nuswide":
            return NuswideMLoader()
        case _:
            print("Unrecognized data set!")
            sys.exit(1)

    image_data = []
    # Go through all images and convert them to Pillow Images
    for image in data:
        image_data.append(functional.to_pil_image(image).convert("RGB"))

    return image_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Based Image Encoder')
    parser.add_argument("-d", "--dataset", type=str, help="dataset to use (cifar-10), sets: cifar-10, mnist, nuswide",
                        default="cifar-10")
    parser.add_argument("-e", "--extract_time", type=bool, default=False, help="Calculate the average time per image")
    parser.add_argument("-c", "--force-cpu", type=bool, default=False, help="Force CPU only")
    args = parser.parse_args()

    if not args.force_cpu:
        if not torch.cuda.is_available():
            raise Exception("No GPU found")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if args.dataset:
        encoder = Encoder(args, device)
        if args.extract_time:
            encoder.calculate_extraction_time()
        else:
            encoder.encode()
    else:
        print("Specify a dataset using flag -d")
