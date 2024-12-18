import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import AlexNet_Weights as AlexNet_Weights
from torchvision.models import ResNet50_Weights as ResNet_Weights


class AlexNet(nn.Module):
    def __init__(self, hash_bit, is_mnist: bool = False, device=torch.device("cuda:0")):
        super(AlexNet, self).__init__()

        # Initializes the alexnet model
        self.is_mnist = is_mnist
        
        # Uses pre-trained weights from ImageNet
        model_alexnet = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        self.features = model_alexnet.features

        if is_mnist:
            self.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)

        # Initializes layers
        cl1 = nn.Linear(256 * 6 * 6, 4096)
        cl1.weight = model_alexnet.classifier[1].weight
        cl1.bias = model_alexnet.classifier[1].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = model_alexnet.classifier[4].weight
        cl2.bias = model_alexnet.classifier[4].bias

        self.hash_layer = nn.Sequential(
            nn.Dropout(),
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Linear(4096, hash_bit),
        )
        self.to(device)
        self.device = device

    def forward(self, x):
        # Defines the forward pass
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.hash_layer(x)
        return x

    # Multi query function for the model
    def query_with_cuda_multi(self, images: torch.Tensor):
        if self.device != "cuda:0":
            self.device = torch.device("cuda:0")
            self.to(self.device)

        if images.shape[0] == 1:
            images = images.squeeze(0)

        self.eval()
        with torch.no_grad():  # Disable gradient calculation for inference
            images = images.to(self.device)  # Move the image to the specified device
            output = self(images)  # Pass through the network
            binary_hash_code = output.data.cpu().sign()  # Convert to binary hash code
        return binary_hash_code

    # Single query function for the model
    def query_single(self, image: torch.Tensor):
        self.eval()
        with torch.no_grad():  # Disable gradient calculation for inference
            image = image.to(self.device)  # Move the image to the specified device
            output = self(image.unsqueeze(0))  # Pass through the network
            binary_hash_code = output.data.cpu().sign()  # Convert to binary hash code
        return binary_hash_code