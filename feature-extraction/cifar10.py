from encode import Encoder
import torch
from feature_loader import FeatureLoader
from imagedata import ImageData

class Cifar10(ImageData):
    def __init__(self, device):
        super().__init__()
        self.dataset_name = "cifar-10"
        self.encoder = Encoder(device)
        self.loader = FeatureLoader(self.dataset_name)

    def getLoader(self):
        return self.loader

    def getEncoder(self):
        return self.encoder

    def encode_dataset(self):
        self.encoder.encode_batches_and_save(self.dataset_name)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE=" + str(device))
    data = Cifar10(device)
    data.encode_dataset()
