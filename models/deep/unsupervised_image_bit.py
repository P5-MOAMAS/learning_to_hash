import time

import torch.nn.functional as F
import torch.optim as optim

from deep_tools.network import *
from deep_tools.tools import *

torch.multiprocessing.set_sharing_strategy('file_system')


# Code gotten from https://github.com/swuxyj/DeepHash-pytorch
# Deep Unsupervised Image Hashing by Maximizing Bit Entropy(AAAI2021)
# paper [Deep Unsupervised Image Hashing by Maximizing Bit Entropy](https://arxiv.org/pdf/2012.12334.pdf)
# code [Deep-Unsupervised-Image-Hashing](https://github.com/liyunqianggyn/Deep-Unsupervised-Image-Hashing)
# [BiHalf Unsupervised] epoch:40, bit:64, dataset:cifar10-2, MAP:0.593, Best MAP: 0.593
def get_config():
    config = {
        "gamma": 6,
        # "optimizer": {"type": optim.SGD, "epoch_lr_decrease": 30, "optim_params": {"lr": 0.0001, "weight_decay": 5e-4, "momentum": 0.9}},
        "optimizer": {"type": optim.RMSprop, "epoch_lr_decrease": 30,
                      "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}, "lr_type": "step"},
        "info": "[BiHalf Unsupervised]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        "net": BiHalfModelUnsupervised,
        # "dataset": "mnist",
        #"dataset": "cifar10-1",
         "dataset": "nuswide_81_m",
        "epoch": 50,
        "test_map": 5,
        "save_path": "save/BiHalf",
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:0"),
        "bit_list": [8],
    }
    config = config_dataset(config)

    config["topK"] = 1000

    return config


class BiHalfModelUnsupervised(nn.Module):
    def __init__(self, bit, is_mnist: bool, device=torch.device("cuda:0")):
        super(BiHalfModelUnsupervised, self).__init__()
        
        #Initializes the alexnet model
        self.alexnet = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        self.alexnet.classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:6])

        #Changes the first layer of the model to accept 1 channel images (for MNIST)
        if is_mnist:
            self.alexnet.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)

        #Freezes the parameters of the model
        for param in self.alexnet.parameters():
            param.requires_grad = False

        #Initializes the fully connected layer
        self.fc_encode = nn.Linear(4096, bit)
        
        #Sets the device
        self.to(device)
        self.device = device

    class Hash(torch.autograd.Function):
        @staticmethod
        def forward(ctx, U):
            # Yunqiang for half and half (optimal transport)
            
            # Sorts the values in U in descending order
            _, index = U.sort(0, descending=True)
            
            # Gets the number of rows and columns in U
            N, D = U.shape
            
            # Creates a binary matrix with half of the values as 1 and the other half as -1
            B_creat = torch.cat((torch.ones([int(N / 2), D]), -torch.ones([N - int(N / 2), D]))).to(config["device"])
            
            # Creates a tensor of zeros with the same shape as U and scatter the values of B_creat in the indices specified by index
            B = torch.zeros(U.shape).to(config["device"]).scatter_(0, index, B_creat)
            
            # Saves the values of U and B for backward pass
            ctx.save_for_backward(U, B)
            return B

        @staticmethod
        def backward(ctx, g):
            # Retrieves the saved values of U and B
            U, B = ctx.saved_tensors
            
            # Calculates an additional gradient term based on the difference between U and B
            add_g = (U - B) / (B.numel())
            
            # Combines the incoming gradient with the additional gradient term
            grad = g + config["gamma"] * add_g
            return grad

    def forward(self, x):
        # Passes the input through the model
        x = self.alexnet.features(x)
        x = x.view(x.size(0), -1)
        x = self.alexnet.classifier(x)

        h = self.fc_encode(x)
        # If the model is not in training mode, return the sign of the output
        if not self.training:
            return h.sign()
        else:
            # If the model is in training mode, calculate the cosine similarity between the hash codes and the features
            b = BiHalfModelUnsupervised.Hash.apply(h)
            target_b = F.cosine_similarity(b[:x.size(0) // 2], b[x.size(0) // 2:])
            target_x = F.cosine_similarity(x[:x.size(0) // 2], x[x.size(0) // 2:])
            
            # Calculate the loss
            loss = F.mse_loss(target_b, target_x)
            return loss

    # Multi Query function for the model
    def query_with_cuda_multi(self, images):
        if self.device != "cuda:0":
            self.device = torch.device("cuda:0")
            self.to(self.device)

        self.eval()
        with torch.no_grad():  # Disable gradient calculation for inference
            images = images.to(torch.device("cuda:0"))  # Move the image to the specified device
            output = self(images)  # Add batch dimension and pass through the network
            binary_hash_code = output.data.cpu().sign()  # Convert to binary hash code
        return binary_hash_code

    # Query function for the model
    def query(self, image):
        self.eval()
        with torch.no_grad():  # Disable gradient calculation for inference
            image = image.to(self.device)  # Move the image to the specified device
            output = self(image.unsqueeze(0))  # Add batch dimension and pass through the network
            binary_hash_code = output.data.cpu().sign()  # Convert to binary hash code
        return binary_hash_code


def train_val(config, bit):
    # Initialization of training
    start_time = time.time()
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit, config["dataset"] == "mnist").to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    Best_mAP = 0
    for epoch in range(config["epoch"]):

        lr = config["optimizer"]["optim_params"]["lr"] * (0.1 ** (epoch // config["optimizer"]["epoch_lr_decrease"]))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, lr:%.9f, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, lr, config["dataset"]), end="")

        # Set the network to training mode
        net.train()
        
        train_loss = 0
        pass_time = 0
        
        for image, _, ind in train_loader:
            # Move the image to the specified device
            image = image.to(device)

            # Clear the gradients
            optimizer.zero_grad()
            
            # Pass the image through the network to calculate loss
            loss = net(image)
            
            # Add up the loss
            train_loss += loss.item()

            # Backpropagate the loss
            loss.backward()
            optimizer.step()

        # Calculate the average loss
        train_loss = train_loss / len(train_loader)
        
        minutes, seconds = divmod(time.time() - start_time, 60)
        hours, minutes = divmod(minutes, 60)
        elapsed_time = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
        print("Total time elapsed: " + elapsed_time)

        print("\b\b\b\b\b\b\b loss:%.9f" % (train_loss))

        if (epoch + 1) % config["test_map"] == 0:
            Best_mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)


if __name__ == "__main__":
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        train_val(config, bit)
