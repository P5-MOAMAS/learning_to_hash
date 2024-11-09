import argparse
import torch
from feature_extraction.main import Encoder
import sys

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
