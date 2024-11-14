import sys
from PIL import Image
import pickle

from torchvision import datasets
from torchvision.transforms import transforms, ToTensor, functional
from torch.utils.data import Dataset
import torch

from feature_extraction.image_net_loader import ImageNetLoader
from util.progressbar import progressbar

class DynamicDataset(Dataset):
    def __init__(self, dataset_name: str, start_index: int = 1):
        self.start_index = start_index
        self.batch = start_index
        self.name = dataset_name
        self.imagenet_loader = None

        match dataset_name:
            case "cifar-10":
                self.max_batch = 5
                self.func = self.__load_cifar10__
            case "mnist":
                self.max_batch = 6
                self.func = self.__load_mnist__
            case "image-net":
                self.max_batch = 55
                self.func = self.__load_image_net__
                self.imagenet_loader = ImageNetLoader()
            case _:
                print("No dataset with given name!")
                sys.exit(1)


    def __next__(self) -> list | None:
        self.batch += 1
        return self.func()


    def __getitem__(self, index: int) -> list | None:
        self.batch = index
        return self.func()


    def __len__(self) -> int:
        return self.max_batch - self.start_index


    def get_name(self) -> str:
        return self.name


    def get_batch_number(self) -> int:
        return self.batch


    def get_max_batch(self) -> int:
        return self.max_batch


    def transform_images(self, images: list) -> list:
        trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        print("Transforming images")
        transformed_images = []
        for i in progressbar(images):
            transformed_images.append(trans(i).unsqueeze(0))
        return transformed_images


    def cifar_to_image(self, data) -> list:
        print("Converting bytes to image")
        images = []
        for d in progressbar(data):
            images.append(Image.frombytes("RGB", (32, 32), d))
        return images


    def mnist_to_image(self, data: torch.Tensor) -> list:
        print("Converting tensor to image")
        images = []
        for t in progressbar(data):
            images.append(functional.to_pil_image(t).convert("RGB"))
        return images


    def check_bounds(self) -> None:
        if self.batch > self.max_batch:
            print("Batch number does not exist!")
            sys.exit(1)
        elif self.batch < self.start_index:
            print("Batch number does not exist!")
            sys.exit(1)


    def __load_cifar10__(self) -> list:
        self.check_bounds()

        with open("cifar-10-batches-py/data_batch_" + str(self.batch), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')

        return self.transform_images(self.cifar_to_image(dict[b'data']))


    def __load_mnist__(self) -> list:
        mnist = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )
        batch_start = (self.batch - 1) * 10000
        data = mnist.data[batch_start:batch_start + 10000]

        return self.transform_images(self.mnist_to_image(data))


    def __load_image_net__(self):
        self.check_bounds()
        print("Loading ImageNet dataset")
        batch_start = (self.batch - 1) * 10000
        batch_size = 10000 if self.batch != self.max_batch else 4546
        return self.transform_images(self.imagenet_loader.load_images(batch_start, batch_start + batch_size))

if __name__ == "__main__":
    dd = DynamicDataset("image-net")
    data = dd[54]
    print(len(data))
