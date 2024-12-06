import torch
from torchvision import transforms

from models.deep.unsupervised_image_bit import BiHalfModelUnsupervised
from utility.data_loader import Dataloader
from utility.metrics_framework import MetricsFramework

################################################################
############################ Config ############################
################################################################
k = [100, 250, 500, 1000, 2500, 5000]
query_size = 10000
encode_length = 64
dataset_name = "cifar-10"
model_path = "saved_models/BiHalf-Cifar10/BiHalf_64bits_0.489/model.pt"
device = torch.device("cuda:0")
is_mnist = False

if is_mnist:
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize([0.5], [0.5])
    ])
else:
    trans = transforms.Compose([])

# Initialize the AlexNet model with x bits
bihalf = BiHalfModelUnsupervised(encode_length).to(device)

# Load the model from the saved state
bihalf.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))

fl = Dataloader(dataset_name)

# Calculate the metrics
metrics_framework = MetricsFramework(bihalf.query_with_cuda_multi, fl, query_size, trans=trans, multi_encoder=True)
mAP = metrics_framework.calculate_metrics(dataset_name + "/bihalf_" + str(encode_length) + "_bits_" + dataset_name, k)
