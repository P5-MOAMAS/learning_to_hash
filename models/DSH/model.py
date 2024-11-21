import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """
        Residual Block used in the ResNet architecture. This block consists of two convolutional layers,
        batch normalization, and a ReLU activation.
    """

    def __init__(self, in_channels: int, out_channels: int, stride=1):
        super().__init__()
        # Define the main convolutional layers for the residual block
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), # Batch normalization
            nn.ReLU(inplace=True), # ReLU activation function
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),  # Batch normalization again
        )

        # Determine if downsampling is needed (when stride is not 1 or input/output channels are different)
        self.do_downsample = stride != 1 or in_channels != out_channels
        if self.do_downsample:
            # If downsampling is required, add a 1x1 convolution layer to match output dimensions
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            # If no downsampling, use an identity layer (no change to the input)
            self.downsample_layer = nn.Identity()

        # Initialize the weights of the layers in the network
        self.apply(self.init_weights)

    def forward(self, x):
        # Store the input
        identity = x
        # Pass input through the network
        out = self.net(x)
        # If downsampling is needed, apply the downsampling layer to the input
        identity = self.downsample_layer(x) if self.do_downsample else identity
        # Add the identity (input) to the output and apply ReLU activation
        return F.relu(out + identity, inplace=True)

    @staticmethod
    def init_weights(m):
        # Initialize the weights using Xavier
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

class ResNet(nn.Module):
    """
        ResNet model with a few residual blocks. This network reduces the dimensionality of the input
        and finally outputs predictions
    """
    def __init__(self, channels: int, size: int, num_classes: int):
        super().__init__()
        # Define the layers of the ResNet, made up of residual blocks
        self.net = nn.Sequential(
            ResBlock(in_channels=channels, out_channels=16),
            ResBlock(in_channels=16, out_channels=16),
            ResBlock(in_channels=16, out_channels=16, stride=2),
        )

        # Compute the size of the input to the linear layer
        self.linear_input_size = self._get_linear_input_size((channels, size, size))
        # Fully connected layer (linear) for classification, from the final feature map to the number of classes
        self.linear = nn.Linear(self.linear_input_size, num_classes)
        # Apply weight initialization to the network
        self.apply(self.init_weights)

    def _get_linear_input_size(self, input_shape):
        # Pass a dummy tensor to calculate the size after convolutions
        with torch.no_grad():
            x = torch.randn(1, *input_shape)
            x = self.net(x)
            return x.numel()

    def forward(self, x):
        # Pass the input through the residual blocks
        x = self.net(x)
        # Flatten the output of the convolutional layers to feed it into the linear layer
        x = x.view(x.size(0), -1)
        # Pass the flattened output through the linear layer and return
        return self.linear(x)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

class LiuDSH(nn.Module):
    """
        LiuDSH model for Deep Supervised Hashing. It uses ResNet and applies
        a linear layer to produce binary hash codes of the specified size.
    """
    def __init__(self, code_size: int, channels: int, size: int, num_classes: int):
        super().__init__()
        # Initialize a ResNet model
        resnet = ResNet(channels= channels, size=size, num_classes=num_classes)
        # Ensure the final linear layer of ResNet will output a binary code of length code_size
        resnet.linear = nn.Linear(in_features=resnet.linear_input_size, out_features=code_size)
        # Use the modified ResNet as the base network for LiuDSH
        self.net = resnet
        # Initialize the weights of the network
        self.apply(self.init_weights)

    def forward(self, x):
        # Pass input through the network
        return self.net(x)

    def create_dummy_tensor(self):
        # Create a dummy tensor based on the initialized values
        return torch.randn((self.num_classes, self.channels, self.size, self.size))

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

    def query_image(self, image: torch.Tensor):
        "Computes and returns a binary hashcode for 1 image"

        self.eval()
        with torch.no_grad():
            embedding = self.forward(image.unsqueeze(0).to("cuda"))
            binary_hash = torch.round(embedding.clamp(-1,1) * 0.5 + 0.5).cpu().int()
            return binary_hash.float()

if __name__ == '__main__':
    # Create a dummy tensor to simulate the input data
    dummy_tensor = LiuDSH.create_dummy_tensor()
    # Initialize the LiuDSH model with a code size of 11
    dsh = LiuDSH(code_size=11)
    print(dsh)
    print(dsh(dummy_tensor).size())
