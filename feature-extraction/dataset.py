import sys
import cifar10

from torch.utils.data import Dataset

class DynamicDataset(Dataset):
    def __init__(self, dataset_name: str, batch_number=0):
        self.datalist = []

        match dataset_name:
            case "cifar-10":
                self.datalist = cifar10.load_cifar10(batch_number)

            case _:
                print("No dataset with given name!")
                sys.exit(1)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        return self.datalist[idx]
