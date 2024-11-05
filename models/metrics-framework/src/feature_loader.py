import sys
from dataloader import DataLoader

class FeatureLoader(DataLoader):
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

        match dataset_name:
            case "cifar-10":
                self.func = self.__get_cifar_features__
                self.batch_len = 5
            case _:
                print("Unrecognized feature set!")
                sys.exit(1)

    def __get_cifar_features__(self, idx):
        name = "features/" + str(self.dataset_name) + "-" + str(idx + 1) + "-features"
        return self.get_data(name)