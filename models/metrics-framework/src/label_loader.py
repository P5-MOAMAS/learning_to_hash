import sys
from dataloader import DataLoader

class LabelLoader(DataLoader):
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

        match dataset_name:
            case "cifar-10":
                self.func = self.__get_cifar_label__
                self.batch_len = 5
            case _:
                print("Unrecognized feature set!")
                sys.exit(1)

    def __get_cifar_label__(self, idx):
        name = "label/" + str(self.dataset_name) + "-" + str(idx + 1) + "-labels"
        return self.get_data(name)