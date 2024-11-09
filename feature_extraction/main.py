# External dependencies
import argparse
import torch
from torch import nn
import sys
import os
import gc

# Internal dependencies
from feature_extraction.vgg import get_configs, VGGAutoEncoder
from util.progressbar import progressbar
from feature_extraction.dataset import DynamicDataset


class Encoder:
    def __init__(self, device):
        self.device = device
        self.model = self.load_model()


    def encode_batch_images(self, batch: list, max_images: int = -1):
        if max_images > 0:
            batch = batch[:max_images]

        print("Starting encoding for " + str(len(batch)) + " images")

        for img in progressbar(batch):
            img = img.to(self.device)
            code = self.encode(img)
            yield code


    def encode_dataset(self, dataset: DynamicDataset):
        for i in range(dataset.get_batch_number(), dataset.get_max_batch() + 1):
            batch = dataset[i]

            if (batch == None):
                sys.exit("Error reading dataset " + dataset.get_name())

            print("Loaded batch " + str(dataset.get_batch_number()) + "/" + str(dataset.get_max_batch()) + " of " + dataset.get_name())
            codes = self.encode_batch(batch)
            print("Finished encoding for batch " + str(dataset.get_batch_number()) + " of " + dataset.get_name() + " structure: " + str(codes.shape))
            yield codes

            # Delete the dataset as soon as the images have been encoded! Saves memory.
            del batch
            gc.collect()


    def encode_batches(self, dataset_name: str):
        dataset = DynamicDataset(dataset_name)
        yield self.encode_dataset(dataset)


    def encode_batch(self, dataset: list):
        codes = []
        for code in self.encode_batch_images(dataset):
            codes.append(nn.functional.adaptive_avg_pool2d(code, (1, 1)).squeeze())

        return torch.stack(codes, dim=0)


    def encode_batches_and_save(self, dataset_name: str):
        dataset = DynamicDataset(dataset_name)
        for codes in self.encode_dataset(dataset):
            os.makedirs("features", exist_ok=True)
            file = "features/" + str(dataset_name) + "-" + str(dataset.get_batch_number()) + "-features"
            print("Saving features to " + file)
            torch.save(codes, file)
            print("Features successfully saved!\n")

            del codes


    def load_model(self):
        model = VGGAutoEncoder(get_configs())
        model = nn.DataParallel(model).to(self.device)

        self.load_dict("saved_models/imagenet-vgg16.pth", model)
        model.eval()

        return model


    def load_dict(self, path: str, model: nn.Module):
        if os.path.isfile(path):
            if self.device.type == 'cuda':
                checkpoint = torch.load(path, weights_only=False)
            else:
                checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
            model_dict = model.state_dict()
            model_dict.update(checkpoint['state_dict'])
            model.load_state_dict(model_dict)
            del checkpoint
        else:
            sys.exit("No model found on path: " + path)
        return model


    def encode(self, img):
        with torch.no_grad():
            code = self.model.module.encoder(img).cpu()
        return code

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Based Image Encoder')
    parser.add_argument("-d", "--dataset", type=str, help="dataset to use (cifar-10), sets: cifar-10, mnist", default="cifar-10")
    parser.add_argument("-fc", "--force-cpu", action="store_true", help="force cpu")
    parser.add_argument("-fg", "--force-gpu", action="store_true", help="force gpu")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.force_cpu:
        args.device = torch.device("cpu")
    elif args.force_gpu:
        if torch.cuda.is_available():
            args.device = torch.device("cuda")
        else:
            sys.exit("No GPU found")

    encoder = Encoder(args.device)

    if args.dataset:
        encoder.encode_batches_and_save(args.dataset)
    else:
        sys.exit("Specify a dataset with -d <dataset name>")
