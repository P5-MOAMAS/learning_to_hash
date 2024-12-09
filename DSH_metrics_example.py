from models.DSH.train import *
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
    (8, "saved_models/DSH-MNIST/MNIST_8bits_0.96/model.pt"),
    (16, "saved_models/DSH-MNIST/MNIST_16bits_0.974/model.pt"),
    (32, "saved_models/DSH-MNIST/MNIST_32bits_0.982/model.pt"),
    (64, "saved_models/DSH-MNIST/MNIST_64bits_0.989/model.pt")
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
    print(f"\nCalculating metrics for DSH at {encode_length} bits")
    dsh = AlexNet(encode_length, is_mnist).to(torch.device("cuda:0"))
    dsh.load_state_dict(torch.load(model_path, map_location=torch.device("cuda:0"), weights_only=False))
    dsh.eval()

    # Calculate the metrics
    metrics_framework = MetricsFramework(dsh.query_with_cuda_multi, fl, query_size, trans=trans, multi_encoder=True)
    metrics_framework.calculate_metrics(dataset_name + "/dsh_" + str(encode_length) + "_bits_" + dataset_name, k)
