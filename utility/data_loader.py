import sys

from torchvision import datasets
from torchvision.transforms import ToTensor, functional

from utility.nuswide_loader import NuswideMLoader


class Dataloader:
    def __init__(self, dataset_name: str):
        super().__init__()

        self.labels = None
        self.data = None

        self.dataset_name = dataset_name
        # Set the loading function for the corresponding dataset
        match dataset_name:
            case "cifar-10":
                self.load_func = self.__init_cifar__
            case "mnist":
                self.load_func = self.__init_mnist__
            case "nuswide":
                self.load_func = self.__init_nuswide__
            case _:
                print("Unrecognized data set!")
                sys.exit(1)
        self.load_data_set()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __init_nuswide__(self):
        loader = NuswideMLoader()
        return loader, loader.labels

    """
    Used to load MNIST Dataset
    """
    @staticmethod
    def __init_mnist__():
        mnist = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor()
        )

        # Load all labels
        image_labels = []
        for labels in mnist.targets.numpy():
            image_labels.append(labels)

        # Load all images and convert them to Pillow Images
        image_data = []
        for image in mnist.data.numpy():
            image_data.append(functional.to_pil_image(image))

        return image_data, image_labels

    """
    Used to load Cifar-10
    """
    @staticmethod
    def __init_cifar__():
        cifar = datasets.CIFAR10(root="data", train=True, download=True)

        # Load all labels
        image_labels = []
        for labels in cifar.targets:
            image_labels.append(labels)

        # Load all images and convert them to Pillow Images
        image_data = []
        for image in cifar.data:
            image_data.append(functional.to_pil_image(image).convert("RGB"))

        return image_data, image_labels

    """
    Loads a dataset using the given load function
    """
    def load_data_set(self):
        image_data, image_labels = self.load_func()

        # Check if the length of image data matches the length of labels
        if len(image_data) != len(image_labels):
            raise Exception("Not as many labels as there are images, data size:", len(image_data), "label size:",
                            len(image_labels))

        self.data = image_data
        self.labels = image_labels


if __name__ == '__main__':
    fl = Dataloader("nuswide")
    if fl.data is None:
        raise Exception("Loaded data is null")
    print(len(fl.data))
