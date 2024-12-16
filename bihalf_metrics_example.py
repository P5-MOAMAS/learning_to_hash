from enum import Enum

import torch
from torchvision import transforms

from models.deep.unsupervised_image_bit import BiHalfModelUnsupervised
from utility.data_loader import Dataloader
from utility.metrics_framework import calculate_encoding_time, save_results, MetricsFramework


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

saved_models = [
    [(8, "saved_models/BiHalf-MNIST/BiHalf_8bits_0.211/model.pt"),
     (16, "saved_models/BiHalf-MNIST/BiHalf_16bits_0.248/model.pt"),
     (32, "saved_models/BiHalf-MNIST/BiHalf_32bits_0.333/model.pt"),
     (64, "saved_models/BiHalf-MNIST/BiHalf_64bits_0.343/model.pt")],
    [(8, "saved_models/BiHalf-Cifar10/BiHalf_8bits_0.301/model.pt"),
     (16, "saved_models/BiHalf-Cifar10/BiHalf_16bits_0.375/model.pt"),
     (32, "saved_models/BiHalf-Cifar10/BiHalf_32bits_0.432/model.pt"),
     (64, "saved_models/BiHalf-Cifar10/BiHalf_64bits_0.489/model.pt")],
    [(8, "saved_models/BiHalf-NUSWIDE81/BiHalf_8bits_0.529/model.pt"),
     (16, "saved_models/BiHalf-NUSWIDE81/BiHalf_16bits_0.583/model.pt"),
     (32, "saved_models/BiHalf-NUSWIDE81/BiHalf_32bits_0.646/model.pt"),
     (64, "saved_models/BiHalf-NUSWIDE81/BiHalf_64bits_0.693/model.pt")]
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
        print(f"\nCalculating metrics for Bihalf at {encode_length} bits")
        # Initialize the AlexNet model with x bits
        bihalf_gpu = BiHalfModelUnsupervised(encode_length, dataset == Dataset.MNIST, device=torch.device("cuda:0"))
        bihalf_gpu.load_state_dict(torch.load(model_path, map_location=torch.device("cuda:0"), weights_only=False))

        bihalf_cpu = BiHalfModelUnsupervised(encode_length, dataset == Dataset.MNIST, device=torch.device("cpu"))
        bihalf_cpu.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"), weights_only=False))

        # Calculate the metrics
        metrics_framework = MetricsFramework(bihalf_gpu.query_with_cuda_multi, fl, query_size, trans=trans, multi_encoder=True)
        mAP = metrics_framework.calculate_metrics(dataset_name + "/bihalf_" + str(encode_length) + "_bits_" + dataset_name, k)

        gpu_mean, cpu_mean, processing_mean = calculate_encoding_time(encode_length, fl, trans, bihalf_gpu.query, bihalf_cpu.query)
        bit_times = {"gpu": gpu_mean, "cpu": cpu_mean, "processing": processing_mean, "total_gpu": gpu_mean + processing_mean, "total_cpu": cpu_mean + processing_mean}
        encoding_times[encode_length] = bit_times
    save_results(encoding_times, dataset_name + "/bihalf_encoding_times")
