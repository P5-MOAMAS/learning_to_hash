import sys
from typing import List

import numpy as np
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

from utility.nuswide_loader import NuswideMLoader


class FeatureLoader:
    def __init__(self, dataset_name: str):
        super().__init__()

        self.labels = None
        self.data = None

        self.dataset_name = dataset_name
        match dataset_name:
            case "cifar-10":
                self.load_func = self.__init_cifar__
            case "mnist":
                self.load_func = self.__init_mnist__
            case "nuswide":
                self.load_func = self.__init_nuswide__
            case _:
                print("Unrecognized feature set!")
                sys.exit(1)
        self.load_feature_set()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def load_features(self, batch_size: int = 1) -> List:
        image_features = []

        for i in range(1, batch_size + 1):
            name = f"features/{self.dataset_name}-{i}-features"
            try:
                with open(name, 'rb') as f:
                    data = torch.load(f, weights_only=False)
            except FileNotFoundError:
                print(f"File {name} not found!")
                sys.exit(1)

            # Convert data to np array
            data = [np.array(i) for i in data]

            if data is None:
                print(f"No data was retrieved from '{name}'")
                sys.exit(1)

            image_features += data
        return image_features

    def __init_nuswide__(self):
        nuswide = NuswideMLoader()
        return self.load_features(), np.asarray(nuswide.labels)

    def __init_mnist__(self):
        mnist = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )

        # Get all features from all images in MNIST
        image_labels = []
        for labels in mnist.targets.numpy():
            image_labels.append(labels)

        # Get all labels in MNIST
        image_features = self.load_features()
        del mnist
        return image_features, image_labels

    def __init_cifar__(self):
        # Get all features from all images in CIFAR
        image_features = self.load_features()

        # Get all labels in CIFAR
        cifar = datasets.CIFAR10(root="data", train=True, download=True)

        image_labels = []
        for labels in cifar.targets:
            image_labels.append(labels)

        return image_features, image_labels

    def load_feature_set(self):
        image_features, image_labels = self.load_func()
        # Check if the length of features matches the length of labels
        if len(image_features) != len(image_labels):
            raise Exception("Not as many labels as there are images, feature size:", len(image_features), "label size:",
                            len(image_labels))

        self.labels = image_labels
        self.data = image_features


if __name__ == '__main__':
    fl = FeatureLoader("cifar-10")
    if fl.data is None:
        raise Exception("Loaded data is null")

    print("Training:", len(fl.data))
