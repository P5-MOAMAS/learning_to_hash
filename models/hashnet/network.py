import torch.nn as nn
from torchvision import models
from tqdm import tqdm
import torch
from torchvision.models import AlexNet_Weights as AlexNet_Weights
from torchvision.models import ResNet50_Weights as ResNet_Weights

class AlexNet(nn.Module):
    def __init__(self, hash_bit):
        super(AlexNet, self).__init__()

        model_alexnet = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        self.features = model_alexnet.features
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

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.hash_layer(x)
        return x
    
    def query_with_cuda(self, image):
        self.eval()
        with torch.no_grad():  # Disable gradient calculation for inference
            image = image.to(torch.device("cuda:0"))  # Move the image to the specified device
            output = self(image.unsqueeze(0))  # Add batch dimension and pass through the network
            binary_hash_code = output.data.cpu().sign()  # Convert to binary hash code
        return binary_hash_code
    
    def query_with_cpu(self, image):
        self.eval()
        with torch.no_grad():  # Disable gradient calculation for inference
            image = image.to(torch.device("cuda:0"))  # Move the image to the specified device
            output = self(image.unsqueeze(0))  # Add batch dimension and pass through the network
            binary_hash_code = output.data.cpu().sign()  # Convert to binary hash code
        return binary_hash_code



resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34, "ResNet50": models.resnet50,
               "ResNet101": models.resnet101, "ResNet152": models.resnet152}


class ResNet(nn.Module):
    def __init__(self, hash_bit, res_model="ResNet50"):
        super(ResNet, self).__init__()
        model_resnet = resnet_dict[res_model](weights=ResNet_Weights.IMAGENET1K_V1)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        y = self.hash_layer(x)
        return y
    
    def query_with_cuda(self, image):
        self.eval()
        with torch.no_grad():  # Disable gradient calculation for inference
            image = image.to(torch.device("cuda:0"))  # Move the image to the specified device
            output = self(image.unsqueeze(0))  # Add batch dimension and pass through the network
            binary_hash_code = output.data.cpu().sign()  # Convert to binary hash code
        return binary_hash_code
    
    def query_with_cpu(self, image):
        self.eval()
        with torch.no_grad():  # Disable gradient calculation for inference
            image = image.to(torch.device("cpu"))  # Move the image to the specified device
            output = self(image.unsqueeze(0))  # Add batch dimension and pass through the network
            binary_hash_code = output.data.cpu().sign()  # Convert to binary hash code
        return binary_hash_code
