from os import sep
import sys
import psutil
import torch
import numpy as np
from dataloader import DataLoader
from hash_lookup import pre_gen_hash_codes


class FeatureLoader:
    def __init__(self, dataset_name: str):
        super().__init__()

        self.training = None
        self.test = None
        self.validation = None

        self.dataset_name = dataset_name

        match dataset_name:
            case "cifar-10":
                self.__init_cifar__()
            case _:
                print("Unrecognized feature set!")
                sys.exit(1)

    def get_training_features(self):
        return self.training
    
    def get_validation_features(self):
        return self.validation
    
    def get_test_features(self):
        return self.test


    def __init_cifar__(self):
        total = []
        for i in range(1, 6):
            name = "features/cifar-10-" + str(i) + "-features"

            with open(name, 'rb') as f:
                data = torch.load(f)

            data = [np.array(i) for i in data]

            if data is None:
                print("No data was retrieved from '" + name + "'")
                sys.exit(1)
            total += data

        # Arbitrary slicing of array

        slice_len = len(total) // 3
        self.training = total[0:slice_len]
        self.test = total[slice_len:slice_len * 2]
        self.validation = total[slice_len * 2:]


def gen_key(some):
    return 0


if __name__ == '__main__':
    process = psutil.Process()
    mem_before = process.memory_info().rss
    
    fl = FeatureLoader("cifar-10")
    print("Training:", len(fl.get_training_features()), sep=" ")
    print("Validation:", len(fl.get_validation_features()), sep=" ")
    print("Test:", len(fl.get_test_features()), sep=" ")

    d = pre_gen_hash_codes(gen_key, fl.get_training_features())

    print(d[0])

    mem_after = process.memory_info().rss
    print("Before: " + str(mem_before))
    print("After: " + str(mem_after))
    print("Delta: " + str(mem_after - mem_before))

