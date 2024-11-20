import pickle
import sys

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms, functional


class Dataloader:
    def __init__(self, dataset_name: str):
        super().__init__()

        self.batch = 1

        self.dataset_name = dataset_name
        match dataset_name:
            case "cifar-10":
                self.load_func = self.__init_cifar__
                self.length = 5
            case "mnist":
                self.load_func = self.__init_mnist__
                self.length = 6
            case _:
                print("Unrecognized data set!")
                sys.exit(1)


    def __getitem__(self, index: int) -> list | None:
        self.batch = index
        return self.load_data_set()


    def __len__(self):
        return self.length


    def __next__(self) -> list | None:
        self.batch += 1
        return self.load_data_set()


    def transform_images(self, images: list) -> list:
        trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        print("Transforming images")
        transformed_images = []
        for i, img in enumerate(images):
            transformed_images.append(trans(img).unsqueeze(0))
        return transformed_images


    def mnist_to_image(self, data: torch.Tensor) -> list:
        print("Converting tensor to image")
        images = []
        for i, t in enumerate(data):
            images.append(functional.to_pil_image(t).convert("RGB"))
        return images


    def __init_mnist__(self):
        mnist = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )

        batch_start = (self.batch - 1) * 10000
        image_labels = mnist.targets[batch_start:batch_start + 10000]
        data = mnist.data[batch_start:batch_start + 10000]

        return self.transform_images(self.mnist_to_image(data)), image_labels

    # @TODO redo
    def __init_cifar__(self):
        # Get all features from all images in CIFAR
        image_features = []
        image_labels = []

        # Get all labels in CIFAR
        fp = "cifar-10-batches-py/"
        file_names = [fp + f"data_batch_{i}" for i in range(1, 6)]

        for name in file_names:
            labels = unpickle(name)[b'labels']
            image_labels += labels

        return image_features, image_labels


    def load_data_set(self):
        image_data, image_labels = self.load_func()
        # Check if the length of features matches the length of labels
        if len(image_data) != len(image_labels):
            raise Exception("Not as many labels as there are images, feature size:", len(image_data), "label size:", len(image_labels))

        # Create index for features and labels
        return [((id + (self.batch - 1) * 10000), feature, label) for id, (feature, label) in
                             enumerate(zip(image_data, image_labels))]


def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


if __name__ == '__main__':
    def gen_key(some):
        return 0

    fl = Dataloader("mnist")
    if fl.training == None or fl.validation == None or fl.test == None:
        raise Exception("One of the sets were null")

    print("Training:", len(fl.training), sep=" ")
    print("Validation:", len(fl.validation), sep=" ")
    print("Test:", len(fl.test), sep=" ")
