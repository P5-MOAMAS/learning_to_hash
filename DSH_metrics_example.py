from enum import Enum

from models.DSH.train import *
from models.deep.deep_tools.network import AlexNet
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
    [(8, "saved_models/DSH-MNIST/MNIST_8bits_0.96/model.pt"),
     (16, "saved_models/DSH-MNIST/MNIST_16bits_0.974/model.pt"),
     (32, "saved_models/DSH-MNIST/MNIST_32bits_0.982/model.pt"),
     (64, "saved_models/DSH-MNIST/MNIST_64bits_0.989/model.pt")],
    [(8, "saved_models/DSH-Cifar10/Cifar10_8bits_0.716/model.pt"),
     (16, "saved_models/DSH-Cifar10/Cfiar10_16bits_0.755/model.pt"),
     (32, "saved_models/DSH-Cifar10/Cifar10_32bits_0.784/model.pt"),
     (64, "saved_models/DSH-Cifar10/Cifar10_64bits_0.788/model.pt")],
    [(8, "saved_models/DSH-NUSWIDE81/NUSWIDE81_8bits_0.681/model.pt"),
     (16, "saved_models/DSH-NUSWIDE81/NUSWIDE81_16bits_0.707/model.pt"),
     (32, "saved_models/DSH-NUSWIDE81/NUSWIDE81_32bits_0.72/model.pt"),
     (64, "saved_models/DSH-NUSWIDE81/NUSWIDE81_64bits_0.736/model.pt")]
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
        print(f"\nCalculating metrics for DSH at {encode_length} bits")
        dsh_gpu = AlexNet(encode_length, dataset == Dataset.MNIST, device=torch.device("cuda:0"))
        dsh_gpu.load_state_dict(torch.load(model_path, map_location=torch.device("cuda:0"), weights_only=False))
        dsh_gpu.eval()

        dsh_cpu = AlexNet(encode_length, dataset == Dataset.MNIST, device=torch.device("cpu"))
        dsh_cpu.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"), weights_only=False))
        dsh_cpu.eval()

        # Calculate the metrics
        metrics_framework = MetricsFramework(dsh_gpu.query_with_cuda_multi, fl, query_size, trans=trans, multi_encoder=True)
        metrics_framework.calculate_metrics(dataset_name + "/dsh_" + str(encode_length) + "_bits_" + dataset_name, k)

        gpu_mean, cpu_mean, processing_mean = calculate_encoding_time(encode_length, fl, trans, dsh_gpu.query_single, dsh_cpu.query_single)
        bit_times = {"gpu": gpu_mean, "cpu": cpu_mean, "processing": processing_mean, "total_gpu": gpu_mean + processing_mean, "total_cpu": cpu_mean + processing_mean}
        encoding_times[encode_length] = bit_times
    save_results(encoding_times, dataset_name + "/dsh_encoding_times")
