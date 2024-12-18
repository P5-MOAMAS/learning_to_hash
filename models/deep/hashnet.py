import math

import torch.optim as optim

from deep_tools.network import *
from deep_tools.tools import *

torch.multiprocessing.set_sharing_strategy('file_system')


# Code gotten from https://github.com/swuxyj/DeepHash-pytorch
# HashNet(ICCV2017)
# paper [HashNet: Deep Learning to Hash by Continuation](http://openaccess.thecvf.com/content_ICCV_2017/papers/Cao_HashNet_Deep_Learning_ICCV_2017_paper.pdf)
# code [HashNet caffe and pytorch](https://github.com/thuml/HashNet)

def get_config():
    config = {
        "alpha": 0.1,
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}, "lr_type": "step"},
        "info": "[HashNet]",
        "step_continuation": 20,
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        "net": AlexNet,
        # "dataset": "cifar10-1",
        "dataset": "nuswide_81_m",
        # "dataset": "mnist",
        "epoch": 50,
        "test_map": 5,
        "save_path": "save/HashNet",
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:0"),
        "bit_list": [8, 16, 32, 64],
    }
    config = config_dataset(config)
    return config


class HashNetLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(HashNetLoss, self).__init__()
        self.scale = 1

    def forward(self, u, y, ind, config):
        # Applies hyperbolic tangent to u, resulting in values between -1 and 1
        u = torch.tanh(self.scale * u)

        # Calculates the similarity matrix
        S = (y @ y.t() > 0).float()

        # Calculates the dot product of u and its transpose
        sigmoid_alpha = config["alpha"]
        dot_product = sigmoid_alpha * u @ u.t()

        # Creates a mask for the positive values of S, this identifies the positive pairs based on S
        mask_positive = S > 0

        # Creates a mask for the negative values of S, this identifies the negative pairs based on S
        mask_negative = (1 - S).bool()

        # Calculates the negative log probabilities for each pair of samples
        neg_log_probe = dot_product + torch.log(1 + torch.exp(-dot_product)) - S * dot_product

        # Calculates the total number of positive pairs
        S1 = torch.sum(mask_positive.float())

        # Calculates the total number of negative pairs
        S0 = torch.sum(mask_negative.float())

        # Calculates the total number of pairs
        S = S0 + S1

        # Adjusting the negative log probabilities based on the number of positive pairs, so that the contribution of positive pairs to the loss is weighted appropriately
        neg_log_probe[mask_positive] = neg_log_probe[mask_positive] * S / S1

        # Adjusting the negative log probabilities based on the number of negative pairs, so that the contribution of negative pairs to the loss is weighted appropriately
        neg_log_probe[mask_negative] = neg_log_probe[mask_negative] * S / S0

        # Calculates the loss
        loss = torch.sum(neg_log_probe) / S
        return loss


def train_val(config, bit):
    # Initialization of training
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit, config["dataset"] == "mnist").to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    # Initialize a loss criterion
    criterion = HashNetLoss(config, bit)

    Best_mAP = 0
    avg_epochs = []

    for epoch in range(config["epoch"]):
        criterion.scale = (epoch // config["step_continuation"] + 1) ** 0.5

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, scale:%.3f, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"], criterion.scale), end="")

        # Set the network to training mode
        net.train()
        epoch_times = []

        train_loss = 0
        for image, label, ind in train_loader:
            # Move the image and label to the specified device
            image = image.to(device)
            label = label.to(device)

            # Clear the gradients
            optimizer.zero_grad()

            # Pass the image through the network
            u = net(image)

            # Calculate the loss
            loss = criterion(u, label.float(), ind, config)
            train_loss += loss.item()

            # Backpropagate the loss
            loss.backward()
            optimizer.step()
            epoch_times.append((time.time_ns() - net.start_time) * math.pow(10, -6))
        avg_epochs.append(np.sum(epoch_times))

        # Calculate the average loss
        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))

        # if (epoch + 1) % config["test_map"] == 0:
        #     Best_mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)

    print(f"Epochs: {avg_epochs}, Mean: {np.mean(avg_epochs)}ms")
    return np.mean(avg_epochs)


if __name__ == "__main__":
    config = get_config()
    print(config)
    metrics = {}
    for bit in config["bit_list"]:
        config["pr_curve_path"] = f"log/alexnet/HashNet_{config['dataset']}_{bit}.json"
        avg = train_val(config, bit)
        metrics[bit] = avg if config["dataset"] != "nuswide_81_m" else avg / 2
    print(metrics)
