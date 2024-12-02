from deep_tools.tools import *
from deep_tools.network import *

import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import numpy as np
from torchvision.models import AlexNet_Weights as AlexNet_Weights

torch.multiprocessing.set_sharing_strategy('file_system')

# Code gotten from https://github.com/swuxyj/DeepHash-pytorch
# Deep Unsupervised Image Hashing by Maximizing Bit Entropy(AAAI2021)
# paper [Deep Unsupervised Image Hashing by Maximizing Bit Entropy](https://arxiv.org/pdf/2012.12334.pdf)
# code [Deep-Unsupervised-Image-Hashing](https://github.com/liyunqianggyn/Deep-Unsupervised-Image-Hashing)
# [BiHalf Unsupervised] epoch:40, bit:64, dataset:cifar10-2, MAP:0.593, Best MAP: 0.593
def get_config():
    config = {
        "gamma": 6,
        #"optimizer": {"type": optim.SGD, "epoch_lr_decrease": 30, "optim_params": {"lr": 0.0001, "weight_decay": 5e-4, "momentum": 0.9}},
        "optimizer": {"type": optim.RMSprop, "epoch_lr_decrease": 30, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}, "lr_type": "step"},
        "info": "[BiHalf Unsupervised]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        "net": BiHalfModelUnsupervised,
        "dataset": "cifar10-1",  # in paper BiHalf is "Cifar-10(I)"
        "epoch": 100,
        "test_map": 5,
        "save_path": "save/BiHalf",
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:0"),
        "bit_list": [64],
    }
    config = config_dataset(config)

    config["topK"] = 1000

    return config


class BiHalfModelUnsupervised(nn.Module):
    def __init__(self, bit):
        super(BiHalfModelUnsupervised, self).__init__()
        self.alexnet = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        self.alexnet.classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:6])
        for param in self.alexnet.parameters():
            param.requires_grad = False

        self.fc_encode = nn.Linear(4096, bit)

    class Hash(torch.autograd.Function):
        @staticmethod
        def forward(ctx, U):
            # Yunqiang for half and half (optimal transport)
            _, index = U.sort(0, descending=True)
            N, D = U.shape
            B_creat = torch.cat((torch.ones([int(N / 2), D]), -torch.ones([N - int(N / 2), D]))).to(config["device"])
            B = torch.zeros(U.shape).to(config["device"]).scatter_(0, index, B_creat)
            ctx.save_for_backward(U, B)
            return B

        @staticmethod
        def backward(ctx, g):
            U, B = ctx.saved_tensors
            add_g = (U - B) / (B.numel())
            grad = g + config["gamma"] * add_g
            return grad

    def forward(self, x):
        x = self.alexnet.features(x)
        x = x.view(x.size(0), -1)
        x = self.alexnet.classifier(x)

        h = self.fc_encode(x)
        if not self.training:
            return h.sign()
        else:
            b = BiHalfModelUnsupervised.Hash.apply(h)
            target_b = F.cosine_similarity(b[:x.size(0) // 2], b[x.size(0) // 2:])
            target_x = F.cosine_similarity(x[:x.size(0) // 2], x[x.size(0) // 2:])
            loss = F.mse_loss(target_b, target_x)
            return loss
        
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


def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit).to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    Best_mAP = 0

    for epoch in range(config["epoch"]):

        lr = config["optimizer"]["optim_params"]["lr"] * (0.1 ** (epoch // config["optimizer"]["epoch_lr_decrease"]))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, lr:%.9f, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, lr, config["dataset"]), end="")

        net.train()

        train_loss = 0
        for image, _, ind in train_loader:
            image = image.to(device)

            optimizer.zero_grad()

            loss = net(image)

            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b loss:%.9f" % (train_loss))

        if (epoch + 1) % config["test_map"] == 0:
            Best_mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)


if __name__ == "__main__":
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        train_val(config, bit)