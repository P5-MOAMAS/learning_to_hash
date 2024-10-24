import sys
import numpy as np
import torch


class FeatureLoader:
    def __iter__(self):
        self.batch_it = 0
        return self

    def __next__(self):
        if self.batch_len <= self.batch_it:
            raise StopIteration

        x = self.func(self.batch_it)
        self.batch_it += 1
        return x

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

        match dataset_name:
            case "cifar-10":
                self.func = self.__get_cifar_features__
                self.batch_len = 5
            case _:
                print("Unrecognized feature set!")
                sys.exit(1)


    def __getitem__(self, idx):
        return self.func(idx)


    def __get_cifar_features__(self, idx):
        name = "features/" + str(self.dataset_name) + "-" + str(idx + 1) + "-features"
        data = None;

        with open(name, 'rb') as f:
            import pickle
            data = pickle.load(f)

        data = [np.array(i) for i in data]

        if data == None:
            print("No data was retrieved from '" + name + "'")
            sys.exit(1)

        return data


if __name__ == '__main__':
    fl = FeatureLoader("cifar-10")

    for idx, batch in enumerate(fl):
        print("Loaded batch: " + str(idx + 1) + "!")

    fl[0]
    print("Loaded cifar-10 batch 1 by index!")
