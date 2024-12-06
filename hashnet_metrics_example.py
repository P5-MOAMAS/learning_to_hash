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
encode_length = 8
dataset_name = "nuswide"
model_path = "saved_models/HashNet-NUSWIDE81/hashnet_8bits_0.642/model.pt"
is_mnist = False

# Initialize the AlexNet model with x bits
hash_net = AlexNet(encode_length, is_mnist).to(torch.device("cuda:0"))

# Load the model from the saved state
hash_net.load_state_dict(torch.load(model_path, map_location=torch.device("cuda:0"), weights_only=False))
hash_net.eval()

if is_mnist:
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize([0.5], [0.5])
    ])
else:
    trans = transforms.Compose([])

fl = Dataloader(dataset_name, True)

# Calculate the metrics
metrics_framework = MetricsFramework(hash_net.query_with_cuda_multi, fl, query_size, trans=trans, multi_encoder=True)
mAP = metrics_framework.calculate_metrics(dataset_name + "/hashnet_" + str(encode_length) + "_bits_" + dataset_name, k)
