from enum import Enum

import torch
from torchvision import transforms

from models.deep.deep_tools.network import AlexNet
from utility.data_loader import Dataloader
from utility.metrics_framework import save_results, calculate_encoding_time, MetricsFramework


class Dataset(Enum):
    MNIST = 0
    CIFAR10 = 1
    NUSWIDE = 2


################################################################
############################ Config ############################
################################################################
k = [100, 250, 500, 1000, 2500, 5000]
query_size = 10000
datasets = [Dataset.MNIST, Dataset.CIFAR10, Dataset.NUSWIDE]
device = torch.device("cuda:0")

saved_models = [
    [(8, "saved_models/HashNet-MNIST/hashnet_8bits_0.437/model.pt"),
     (16, "saved_models/HashNet-MNIST/hashnet_16bits_0.964/model.pt"),
     (32, "saved_models/HashNet-MNIST/hashnet_32bits_0.975/model.pt"),
     (64, "saved_models/HashNet-MNIST/hashnet_64bits_0.98/model.pt")],
    [(8, "saved_models/HashNet-Cifar10/hashnet_cifar10_8bits_0.429/model.pt"),
     (16, "saved_models/HashNet-Cifar10/hashnet_cifar10_16bit_0.603/model.pt"),
     (32, "saved_models/HashNet-Cifar10/hashnet_cifar10_32bit_0.774/model.pt"),
     (64, "saved_models/HashNet-Cifar10/hashnet_cifar10_64bit_0.793/model.pt")],
    [(8, "saved_models/HashNet-NUSWIDE81/hashnet_8bits_0.642/model.pt"),
     (16, "saved_models/HashNet-NUSWIDE81/hashnet_16bits_0.691/model.pt"),
     (32, "saved_models/HashNet-NUSWIDE81/hashnet_32bits_0.735/model.pt"),
     (64, "saved_models/HashNet-NUSWIDE81/hashnet_64bits_0.771/model.pt")]
]

for dataset in datasets:
    dataset_name = "mnist" if dataset == Dataset.MNIST else "cifar-10" if dataset == Dataset.CIFAR10 else "nuswide"

    if dataset == Dataset.MNIST:
        normalize = transforms.Normalize([0.5], [0.5])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        normalize,
    ])

    fl = Dataloader(dataset_name)

    encoding_times = {}

    for encode_length, model_path in saved_models[dataset.value]:
        print(f"\nCalculating metrics for HashNet at {encode_length} bits")
        # Initialize the AlexNet model with x bits
        hash_net_cpu = AlexNet(encode_length, dataset == Dataset.MNIST, device=torch.device("cpu"))
        hash_net_cpu.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"), weights_only=False))
        hash_net_cpu.eval()

        hash_net_gpu = AlexNet(encode_length, dataset == Dataset.MNIST, device=torch.device("cuda:0"))
        hash_net_gpu.load_state_dict(torch.load(model_path, map_location=torch.device("cuda:0"), weights_only=False))
        hash_net_gpu.eval()

        fl = Dataloader(dataset_name)

        # Calculate the metrics
        metrics_framework = MetricsFramework(hash_net_gpu.query_with_cuda_multi, fl, query_size, trans=trans, multi_encoder=True)
        mAP = metrics_framework.calculate_metrics(dataset_name + "/hashnet_" + str(encode_length) + "_bits_" + dataset_name, k)

        gpu_mean, cpu_mean, processing_mean = calculate_encoding_time(encode_length, fl, trans, hash_net_gpu.query_single, hash_net_cpu.query_single)
        bit_times = {"gpu": gpu_mean, "cpu": cpu_mean, "processing": processing_mean, "total_gpu": gpu_mean + processing_mean, "total_cpu": cpu_mean + processing_mean}
        encoding_times[encode_length] = bit_times

    save_results(encoding_times, dataset_name + "/hashnet_encoding_times")
