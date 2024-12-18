import math
import os
import sys

import torch.optim as optim

if __name__ == "__main__":
    sys.path.append(os.getcwd())
from models.deep.deep_tools.network import *
from models.deep.deep_tools.tools import *

torch.multiprocessing.set_sharing_strategy('file_system')


# DSH(CVPR2016)
# paper [Deep Supervised Hashing for Fast Image Retrieval](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_Deep_Supervised_Hashing_CVPR_2016_paper.pdf)
# code [DSH-pytorch](https://github.com/weixu000/DSH-pytorch)
# code [CV_Project](https://github.com/aarathimuppalla/CV_Project)
# code [DSH_tensorflow](https://github.com/yg33717/DSH_tensorflow)

def get_config():
    config = {
        "alpha": 0.1,
        # "optimizer":{"type":  optim.SGD, "optim_params": {"lr": 0.05, "weight_decay": 10 ** -5}},
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[DSH]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        "net": AlexNet,
        # "dataset": "cifar10-1",
        "dataset": "nuswide_81_m",
        # "dataset": "mnist",
        "epoch": 50,
        "test_map": 5,
        "save_path": "save/DSH",
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:0"),
        "bit_list": [8, 16, 32, 64],
    }
    config = config_dataset(config)
    return config


class DSHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DSHLoss, self).__init__()
        self.m = 2 * bit
        self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"])

    def forward(self, u, y, ind, config):
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        dist = (u.unsqueeze(1) - self.U.unsqueeze(0)).pow(2).sum(dim=2)
        y = (y @ self.Y.t() == 0).float()

        loss = (1 - y) / 2 * dist + y / 2 * (self.m - dist).clamp(min=0)
        loss1 = loss.mean()
        loss2 = config["alpha"] * (1 - u.abs()).abs().mean()

        return loss1 + loss2


def train_val(config, bit):
    start_time = time.time()
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit, config["dataset"] == "mnist").to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    criterion = DSHLoss(config, bit)

    Best_mAP = 0
    avg_epochs = []

    for epoch in range(3):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net.train()
        epoch_times = []

        train_loss = 0
        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            u = net(image)

            loss = criterion(u, label.float(), ind, config)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            epoch_times.append((time.time_ns() - net.start_time) * math.pow(10, -6))
        avg_epochs.append(np.sum(epoch_times))
        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))

        minutes, seconds = divmod(time.time() - start_time, 60)
        hours, minutes = divmod(minutes, 60)
        elapsed_time = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
        print("Total time elapsed: " + elapsed_time)

        if (epoch + 1) % config["test_map"] == 0:
            Best_mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)
    print(f"Epochs: {avg_epochs}, Mean: {np.mean(avg_epochs)}ms")
    return np.mean(avg_epochs)


if __name__ == "__main__":
    config = get_config()
    print(config)
    metrics = {}
    for bit in config["bit_list"]:
        config["pr_curve_path"] = f"log/alexnet/DSH_{config['dataset']}_{bit}.json"
        avg = train_val(config, bit)
        metrics[bit] = avg if config["dataset"] != "nuswide_81_m" else avg / 2
    print(metrics)
