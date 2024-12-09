import torch
from torchvision import transforms

from models.deep.deep_tools.network import AlexNet
from utility.data_loader import Dataloader
from utility.metrics_framework import MetricsFramework

################################################################
############################ Config ############################
################################################################
k = [100, 250, 500, 1000, 2500, 5000]
query_size = 10000
dataset_name = "mnist"
is_mnist = True

saved_models = [
    (8, "saved_models/HashNet-MNIST/hashnet_8bits_0.437/model.pt"),
    (16, "saved_models/HashNet-MNIST/hashnet_16bits_0.964/model.pt"),
    (32, "saved_models/HashNet-MNIST/hashnet_32bits_0.975/model.pt"),
    (64, "saved_models/HashNet-MNIST/hashnet_64bits_0.98/model.pt")
]

if is_mnist:
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize([0.5], [0.5])
    ])
else:
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

for encode_length, model_path in saved_models:
    print(f"\nCalculating metrics for HashNet at {encode_length} bits")
    # Initialize the AlexNet model with x bits
    hash_net = AlexNet(encode_length, is_mnist).to(torch.device("cuda:0"))

    # Load the model from the saved state
    hash_net.load_state_dict(torch.load(model_path, map_location=torch.device("cuda:0"), weights_only=False))
    hash_net.eval()

    fl = Dataloader(dataset_name)

    # Calculate the metrics
    metrics_framework = MetricsFramework(hash_net.query_with_cuda_multi, fl, query_size, trans=trans,
                                         multi_encoder=True)
    mAP = metrics_framework.calculate_metrics(dataset_name + "/hashnet_" + str(encode_length) + "_bits_" + dataset_name,
                                              k)
