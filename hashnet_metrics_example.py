import torch
from torchvision import transforms

from utility.data_loader import Dataloader
from utility.metrics_framework import MetricsFramework
from models.deep.deep_tools.network import AlexNet

"""
YOU SHOULD RUN THIS SCRIPT FROM THE ROOT DIRECTORY OF THE PROJECT
MAKE SURE THAT YOU HAVE A MODEL SAVED
IN ORDER TO CHANGE THE AMOUNT OF BITS THE MODEL USE, YOU WILL HAVE TO CHANGE THE MODEL BEFORE TRAINING OTHERWISE YOU WILL GET AN ERROR
"""
k = 500
query_size = 10000

config = {
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 16,
        # "dataset": "cifar10",
        "dataset": "cifar10-1",
        # "dataset": "cifar10-2",
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:0"),
        "max_images": 16666, # 59000 is default amount of images
    }

# Initialize the AlexNet model with x bits
hashnet_alexnet_cifar10 = AlexNet(config["batch_size"]).to(config["device"])

# Load the model from the saved state
hashnet_alexnet_cifar10.load_state_dict(torch.load("./models/deep/model.pt", map_location=config["device"], weights_only=False))
#hashnet_alexnet_cifar10.eval()

trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(config["crop_size"]),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

fl = Dataloader("cifar-10")

# Calculate the metrics
metrics_framework = MetricsFramework(hashnet_alexnet_cifar10.query_with_cuda, fl.data, fl.labels, query_size, trans=trans)
mAP = metrics_framework.calculate_metrics(k)
print("Results for k =", k, ":", mAP)
