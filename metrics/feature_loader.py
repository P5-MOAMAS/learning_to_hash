import pickle
import sys
from random import shuffle
from typing import List, Callable

import torch
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor


class FeatureLoader:
    def __init__(self, dataset_name: str, split_data: bool = True):
        super().__init__()

        self.training = None
        self.test = None
        self.validation = None

        self.split_data = split_data

        self.dataset_name = dataset_name
        match dataset_name:
            case "cifar-10":
                self.load_func = self.__init_cifar__
            case "mnist":
                self.load_func = self.__init_mnist__
            case _:
                print("Unrecognized feature set!")
                sys.exit(1)
        self.load_feature_set()


    def load_features(self, batch_size: int) -> List:
        image_features = []

        for i in range(1, batch_size + 1):
            name = f"features/{self.dataset_name}-{i}-features"
            try:
                with open(name, 'rb') as f:
                    data = torch.load(f, weights_only=False)
            except FileNotFoundError:
                print(f"File {name} not found!")
                sys.exit(1)

            data = [np.array(i) for i in data]

            if data is None:
                print(f"No data was retrieved from '{name}'")
                sys.exit(1)

            image_features += data
        return image_features


    def __init_mnist__(self):
        mnist = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )

        image_labels = mnist.targets
        image_features = self.load_features(6)
        del mnist
        return image_features, image_labels


    def __init_cifar__(self):
        # Get all features from all images in CIFAR
        image_features = self.load_features(5)
        image_labels = []

        # Get all labels in CIFAR
        fp = "cifar-10-batches-py/"
        file_names = [fp + f"data_batch_{i}" for i in range(1, 6)]

        for name in file_names:
            labels = unpickle(name)[b'labels']
            image_labels += labels

        return image_features, image_labels


    def load_feature_set(self):
        image_features, image_labels = self.load_func()
        # Check if the length of features matches the length of labels
        if len(image_features) != len(image_labels):
            raise Exception("Not as many labels as there are images, feature size:", len(image_features), "label size:", len(image_labels))

        # Create index for features and labels
        idx_feature_label = [(id, feature, label) for id, (feature, label) in
                             enumerate(zip(image_features, image_labels))]

        # Shuffle data before splitting
        shuffle(idx_feature_label)
        if self.split_data:
            # Splitting dataset into 70% training, 15% testing, and 15% validation
            training_len = round(len(idx_feature_label) * 0.7)
            valid_test_len = (len(idx_feature_label) - training_len) // 2

            self.training = idx_feature_label[:training_len]
            self.test = idx_feature_label[training_len:training_len + valid_test_len]
            self.validation = idx_feature_label[training_len + valid_test_len:]
        else:
            self.training = self.test = self.validation = idx_feature_label


def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


if __name__ == '__main__':
    fl = FeatureLoader("mnist")
    if fl.training == None or fl.validation == None or fl.test == None:
        raise Exception("One of the sets were null")

    print("Training:", len(fl.training), sep=" ")
    print("Validation:", len(fl.validation), sep=" ")
    print("Test:", len(fl.test), sep=" ")

