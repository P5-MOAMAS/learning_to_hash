import sys
from PIL import Image
import pickle

from torchvision import datasets
from torchvision.transforms import transforms, ToTensor, functional
from torch.utils.data import Dataset

from progressbar import progressbar

class DynamicDataset(Dataset):
    def __init__(self, dataset_name, start_index: int = 1):
        self.start_index = start_index
        self.batch = start_index
        self.name = dataset_name

        match dataset_name:
            case "cifar-10":
                self.max_batch = 5
                self.func = self.__load_cifar10__
            case "mnist":
                self.max_batch = 6
                self.func = self.__load_mnist__
            case _:
                print("No dataset with given name!")
                sys.exit(1)


    def __next__(self):
        self.batch += 1
        return self.func()


    def __getitem__(self, index):
        self.batch = index
        return self.func()


    def __len__(self):
        return self.max_batch - self.start_index


    def get_name(self):
        return self.name


    def get_batch_number(self):
        return self.batch


    def get_max_batch(self):
        return self.max_batch


    def transform_images(self, images):
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


    def cifar_to_image(self, data):
        print("Converting bytes to image")
        images = []
        for d in progressbar(data):
            images.append(Image.frombytes("RGB", (32, 32), d))
        return images


    def mnist_to_image(self, tensor):
        print("Converting tensor to image")
        images = []
        for t in progressbar(tensor):
            images.append(functional.to_pil_image(t).convert("RGB"))
        return images


    def check_bounds(self):
        if self.batch > self.max_batch:
            print("Batch number does not exist!")
            sys.exit(1)
        elif self.batch < self.start_index:
            print("Batch number does not exist!")
            sys.exit(1)


    def __load_cifar10__(self):
        self.check_bounds()

        with open("cifar-10-batches-py/data_batch_" + str(self.batch), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')

        return self.transform_images(self.cifar_to_image(dict[b'data']))

    def __load_mnist__(self):
        mnist = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )
        batch_start = (self.batch - 1) * 10000
        data = mnist.data[batch_start:batch_start + 10000]

        return self.transform_images(self.mnist_to_image(data))

if __name__ == "__main__":
    dd = DynamicDataset("mnist")
    data = dd[0]
    print(len(data), data[0].shape)