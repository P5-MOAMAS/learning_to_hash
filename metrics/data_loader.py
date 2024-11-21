import pickle
import sys
from random import shuffle
from typing import List

from PIL import Image
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms, functional


class Dataloader:
    def __init__(self, dataset_name: str):
        super().__init__()

        self.data = None

        self.dataset_name = dataset_name
        match dataset_name:
            case "cifar-10":
                self.load_func = self.__init_cifar__
            case "mnist":
                self.load_func = self.__init_mnist__
            case _:
                print("Unrecognized data set!")
                sys.exit(1)
        self.load_data_set()


    @staticmethod
    def cifar_to_tensor(data: List[bytes]) -> List:
        images = []
        length = len(data)
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        for i, d in enumerate(data, start=1):
            images.append(trans(Image.frombytes("RGB", (32, 32), d)))
            print("Converting bytes to image,", i, "of", length, end="\r", flush=True)
        print()

        return images


    # Will be used for imagenet when it's implemented!
    @staticmethod
    def transform_images(images: List) -> List:
        trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        transformed_images = []
        length = len(images)
        for i, img in enumerate(images, start=1):
            transformed_images.append(trans(img).unsqueeze(0))
            print("Transforming image", i, "of", length, end="\r", flush=True)
        print()

        return transformed_images


    @staticmethod
    def __init_mnist__():
        mnist = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )
        image_labels = mnist.targets

        return mnist.data, image_labels


    def __init_cifar__(self):
        # Get all features from all images in CIFAR
        image_data = []
        image_labels = []

        # Get all labels in CIFAR
        fp = "cifar-10-batches-py/"
        file_names = [fp + f"data_batch_{i}" for i in range(1, 6)]

        for file in file_names:
            with open(file, 'rb') as fo:
                data = pickle.load(fo, encoding='bytes')
                image_labels += data[b'labels']
                image_data += self.cifar_to_tensor(data[b'data'])

        return image_data, image_labels


    def load_data_set(self):
        image_data, image_labels = self.load_func()

        # Check if the length of image data matches the length of labels
        if len(image_data) != len(image_labels):
            raise Exception("Not as many labels as there are images, data size:", len(image_data), "label size:", len(image_labels))

        # Create index for images and labels
        idx_image_label = [(idx, image, label) for idx, (image, label) in enumerate(zip(image_data, image_labels))]

        shuffle(idx_image_label)

        self.data = idx_image_label


if __name__ == '__main__':
    fl = Dataloader("mnist")

    print(len(fl.data))
