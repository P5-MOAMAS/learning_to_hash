import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.do_downsample = stride != 1 or in_channels != out_channels
        if self.do_downsample:
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample_layer = nn.Identity()

        self.apply(self.init_weights)

    def forward(self, x):
        identity = x
        out = self.net(x)
        identity = self.downsample_layer(x) if self.do_downsample else identity
        return F.relu(out + identity, inplace=True)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

class ResNet(nn.Module):
    def __init__(self, channels: int, size: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            ResBlock(in_channels=3, out_channels=16),
            ResBlock(in_channels=16, out_channels=16),
            ResBlock(in_channels=16, out_channels=16, stride=2),
        )

        self.linear_input_size = self._get_linear_input_size((channels, size, size))
        self.linear = nn.Linear(self.linear_input_size, num_classes)
        self.apply(self.init_weights)

    def _get_linear_input_size(self, input_shape):
        # Pass a dummy tensor to calculate the size after convolutions
        with torch.no_grad():
            x = torch.randn(1, *input_shape)
            x = self.net(x)
            return x.numel()

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

class LiuDSH(nn.Module):
    def __init__(self, code_size: int, channels: int, size: int, num_classes: int):
        super().__init__()
        resnet = ResNet(channels= channels, size=size, num_classes=num_classes)
        resnet.linear = nn.Linear(in_features=resnet.linear_input_size, out_features=code_size)
        self.net = resnet
        self.apply(self.init_weights)

    def forward(self, x):
        return self.net(x)

    def create_dummy_tensor(self):
        # Create a dummy tensor based on the initialized values
        return torch.randn((self.num_classes, self.channels, self.size, self.size))

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

if __name__ == '__main__':
    dummy_tensor = LiuDSH.create_dummy_tensor()
    dsh = LiuDSH(code_size=11)
    print(dsh)
    print(dsh(dummy_tensor).size())
