from models.DSH.train import *
from models.deep.deep_tools.network import AlexNet
from utility.data_loader import Dataloader
from utility.metrics_framework import MetricsFramework

################################################################
############################ Config ############################
################################################################
k = [100, 250, 500, 1000, 2500, 5000]
query_size = 10000
encode_length = 8
dataset_name = "mnist"
model_path = "saved_models/DSH-MNIST/MNIST_8bits_0.96/model.pt"
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

dsh = AlexNet(encode_length, True).to(torch.device("cuda:0"))
dsh.load_state_dict(torch.load(model_path, map_location=torch.device("cuda:0"), weights_only=False))
dsh.eval()

fl = Dataloader(dataset_name)

# Calculate the metrics
metrics_framework = MetricsFramework(dsh.query_with_cuda_multi, fl, query_size, trans=trans, multi_encoder=True)
metrics_framework.calculate_metrics(dataset_name + "/dsh_" + str(encode_length) + "_bits_" + dataset_name, k)
