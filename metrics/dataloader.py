import sys
from typing import Callable
import numpy as np
import torch


class DataLoader:
    def __iter__(self):
        self.batch_it = 0
        return self

    def __next__(self):
        if self.batch_len <= self.batch_it:
            raise StopIteration

        x = self.func(self.batch_it)
        self.batch_it += 1
        return x

    def __getitem__(self, idx):
        return self.func(idx)


    def get_data(self, path: str):
        with open(path, 'rb') as f:
            data = torch.load(f)

        data = [np.array(i) for i in data]

        if data is None:
            print("No data was retrieved from '" + path + "'")
            sys.exit(1)

        return data
