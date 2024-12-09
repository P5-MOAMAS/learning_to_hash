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
dataset_name = "mnist"
device = torch.device("cuda:0")
is_mnist = True

saved_models = [
    (8, "saved_models/BiHalf-MNIST/BiHalf_8bits_0.211/model.pt"),
    (16, "saved_models/BiHalf-MNIST/BiHalf_16bits_0.248/model.pt"),
    (32, "saved_models/BiHalf-MNIST/BiHalf_32bits_0.333/model.pt"),
    (64, "saved_models/BiHalf-MNIST/BiHalf_64bits_0.343/model.pt")
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

fl = Dataloader(dataset_name)

for encode_length, model_path in saved_models:
    print(f"\nCalculating metrics for Bihalf at {encode_length} bits")
    # Initialize the AlexNet model with x bits
    bihalf = BiHalfModelUnsupervised(encode_length, is_mnist).to(device)

    # Load the model from the saved state
    bihalf.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))

    # Calculate the metrics
    metrics_framework = MetricsFramework(bihalf.query_with_cuda_multi, fl, query_size, trans=trans, multi_encoder=True)
    mAP = metrics_framework.calculate_metrics(dataset_name + "/bihalf_" + str(encode_length) + "_bits_" + dataset_name,
                                              k)
