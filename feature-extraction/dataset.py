import sys
from PIL import Image
import pickle

from torchvision.transforms import transforms
from torch.utils.data import Dataset

class DynamicDataset(Dataset):
    def __init__(self, dataset_name, start_index: int = 1):
        self.start_index = start_index
        self.batch = start_index
        self.name = dataset_name

        match dataset_name:
            case "cifar-10":
                self.max_batch = 5
                self.func = self.__load_cifar10__

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


    def transform_images(self, dict):
        trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        images = [Image.frombytes("RGB", (32, 32), d) for d in dict[b'data']]
        images = [trans(i).unsqueeze(0) for i in images]
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

        return self.transform_images(dict)