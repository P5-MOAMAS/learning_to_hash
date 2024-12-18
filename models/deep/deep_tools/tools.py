import json
import os

import numpy as np
import torch
import torch.utils.data as util_data
import torchvision.datasets as dsets
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import torchvision.datasets as dsets
import os
import json

# Configures the dataset
def config_dataset(config):
    if "cifar" in config["dataset"]:
        config["topK"] = -1
        config["n_class"] = 10
    elif config["dataset"] in ["nuswide_21", "nuswide_21_m"]:
        config["topK"] = 5000
        config["n_class"] = 21
    elif config["dataset"] == "nuswide_81_m":
        config["topK"] = 5000
        config["n_class"] = 81
    elif config["dataset"] == "coco":
        config["topK"] = 5000
        config["n_class"] = 80
    elif config["dataset"] == "imagenet":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "mirflickr":
        config["topK"] = -1
        config["n_class"] = 38
    elif config["dataset"] == "voc2012":
        config["topK"] = -1
        config["n_class"] = 20
    elif config["dataset"] == "mnist":
        config["topK"] = -1
        config["n_class"] = 10

    config["data_path"] = "/dataset/" + config["dataset"] + "/"
    if config["dataset"] == "nuswide_21":
        config["data_path"] = "/dataset/NUS-WIDE/"
    if config["dataset"] in ["nuswide_21_m", "nuswide_81_m"]:
        config["data_path"] = "data/nuswide_81_m/"
    if config["dataset"] == "coco":
        config["data_path"] = "/dataset/COCO_2014/"
    if config["dataset"] == "voc2012":
        config["data_path"] = "/dataset/"
    config["data"] = {
        "train_set": {"list_path": "data/" + config["dataset"] + "/train.txt", "batch_size": config["batch_size"]},
        "database": {"list_path": "data/" + config["dataset"] + "/database.txt", "batch_size": config["batch_size"]},
        "test": {"list_path": "data/" + config["dataset"] + "/test.txt", "batch_size": config["batch_size"]}}
    return config


class ImageList(object):

    def __init__(self, data_path, image_list, transform):
        # Creates a list of images
        self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    # Returns the image at the specified index
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    # Returns the length of the image list
    def __len__(self):
        return len(self.imgs)

# Transforms the image
def image_transform(resize_size, crop_size, data_set):
    if data_set == "train_set":
        step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
    else:
        step = [transforms.CenterCrop(crop_size)]
    return transforms.Compose([transforms.Resize(resize_size)]
                              + step +
                              [transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                               ])


# Custom CIFAR10 and MNIST classes
class MyCIFAR10(dsets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        target = np.eye(10, dtype=np.int8)[np.array(target)]
        return img, target, index


class MyMNIST(dsets.MNIST):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        target = np.eye(10, dtype=np.int8)[np.array(target)]
        return img, target, index

# Loads the CIFAR10 dataset
def cifar_dataset(config):
    batch_size = config["batch_size"]

    train_size = 500
    test_size = 100

    if config["dataset"] == "cifar10-2":
        train_size = 5000
        test_size = 1000

    # Transforms the image
    transform = transforms.Compose([
        transforms.Resize(config["crop_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    cifar_dataset_root = '/dataset/cifar/'
    # Dataset
    # Splits the dataset into training, testing, and database datasets
    train_dataset = MyCIFAR10(root=cifar_dataset_root,
                              train=True,
                              transform=transform,
                              download=True)

    test_dataset = MyCIFAR10(root=cifar_dataset_root,
                             train=False,
                             transform=transform)

    database_dataset = MyCIFAR10(root=cifar_dataset_root,
                                 train=False,
                                 transform=transform)

    # Concatenates the training, testing, and database datasets
    X = np.concatenate((train_dataset.data, test_dataset.data))
    L = np.concatenate((np.array(train_dataset.targets), np.array(test_dataset.targets)))

    first = True

    # Iterates over the labels
    for label in range(10):
        index = np.where(L == label)[0]

        # Shuffles the indices of the labels
        N = index.shape[0]
        perm = np.random.permutation(N)
        index = index[perm]

        # If first, initializes the test, train, and database indices
        if first:
            test_index = index[:test_size]
            train_index = index[test_size: train_size + test_size]
            database_index = index[train_size + test_size:]
        else:
            # Concatenates the test, train, and database indices
            test_index = np.concatenate((test_index, index[:test_size]))
            train_index = np.concatenate((train_index, index[test_size: train_size + test_size]))
            database_index = np.concatenate((database_index, index[train_size + test_size:]))
        first = False

    if config["dataset"] == "cifar10":
        # test:1000, train:5000, database:54000
        pass
    elif config["dataset"] == "cifar10-1":
        # test:1000, train:5000, database:59000
        database_index = np.concatenate((train_index, database_index))
    elif config["dataset"] == "cifar10-2":
        # test:10000, train:50000, database:50000
        database_index = train_index

    # Sets the data and targets of the training, testing, and database datasets
    train_dataset.data = X[train_index]
    train_dataset.targets = L[train_index]
    test_dataset.data = X[test_index]
    test_dataset.targets = L[test_index]
    database_dataset.data = X[database_index]
    database_dataset.targets = L[database_index]

    # Prints the number of images in the training, testing, and database datasets
    print("train_dataset", train_dataset.data.shape[0])
    print("test_dataset", test_dataset.data.shape[0])
    print("database_dataset", database_dataset.data.shape[0])

    # Creates dataloaders for the training, testing, and database datasets
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4)

    database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=4)

    return train_loader, test_loader, database_loader, \
        train_index.shape[0], test_index.shape[0], database_index.shape[0]

# Loads the MNIST dataset
def mnist_dataset(config):
    batch_size = config["batch_size"]

    train_size = 500
    test_size = 100

    # Transforms the image
    transform = transforms.Compose([
        transforms.Resize(config["crop_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    mnist_dataset_root = '/dataset/mnist/'
    # Dataset loading
    train_dataset = MyMNIST(root=mnist_dataset_root,
                            train=True,
                            transform=transform,
                            download=True)

    test_dataset = MyMNIST(root=mnist_dataset_root,
                           train=False,
                           transform=transform)

    database_dataset = MyMNIST(root=mnist_dataset_root,
                               train=False,
                               transform=transform)

    # Concatenates the training, testing, and database datasets
    X = np.concatenate((train_dataset.data, test_dataset.data))
    L = np.concatenate((np.array(train_dataset.targets), np.array(test_dataset.targets)))

    first = True
    # Iterates over the labels
    for label in range(10):
        index = np.where(L == label)[0]

        # Shuffles the indices of the labels
        N = index.shape[0]
        perm = np.random.permutation(N)
        index = index[perm]

        # If first, initializes the test, train, and database indices
        if first:
            test_index = index[:test_size]
            train_index = index[test_size: train_size + test_size]
            database_index = index[train_size + test_size:]
        else:
            # Concatenates the test, train, and database indices
            test_index = np.concatenate((test_index, index[:test_size]))
            train_index = np.concatenate((train_index, index[test_size: train_size + test_size]))
            database_index = np.concatenate((database_index, index[train_size + test_size:]))
        first = False

    if config["dataset"] == "mnist":
        # test:1000, train:5000, database:59000
        database_index = np.concatenate((train_index, database_index))

    # Sets the data and targets of the training, testing, and database datasets
    train_dataset.data = X[train_index]
    train_dataset.targets = L[train_index]
    test_dataset.data = X[test_index]
    test_dataset.targets = L[test_index]
    database_dataset.data = X[database_index]
    database_dataset.targets = L[database_index]

    # Prints the number of images in the training, testing, and database datasets
    print("train_dataset", train_dataset.data.shape[0])
    print("test_dataset", test_dataset.data.shape[0])
    print("database_dataset", database_dataset.data.shape[0])

    # Creates dataloaders for the training, testing, and database datasets
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4)

    database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=4)

    return train_loader, test_loader, database_loader, \
        train_index.shape[0], test_index.shape[0], database_index.shape[0]

# Gets the data
def get_data(config):
    # If the dataset is CIFAR
    if "cifar" in config["dataset"]:
        return cifar_dataset(config)
    # If the dataset is MNIST
    elif "mnist" in config["dataset"]:
        return mnist_dataset(config)

    # Initializes the datasets and data loaders
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]

    # Iterates over the training, testing, and database datasets
    for data_set in ["train_set", "test", "database"]:
        dsets[data_set] = ImageList(config["data_path"],
                                    open(data_config[data_set]["list_path"]).readlines(),
                                    transform=image_transform(config["resize_size"], config["crop_size"], data_set))
        print(data_set, len(dsets[data_set]))
        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                      batch_size=data_config[data_set]["batch_size"],
                                                      shuffle=(data_set == "train_set"), num_workers=4)

    # Returns the training, testing, and database datasets as dataloaders and their sizes
    return dset_loaders["train_set"], dset_loaders["test"], dset_loaders["database"], \
        len(dsets["train_set"]), len(dsets["test"]), len(dsets["database"])


# Computes the binary codes for the specific model
def compute_result(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        bs.append((net(img.to(device))).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)

# Calculates the Hamming distance
def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


# Calculates the top map
def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        # Calculate the ground truth
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)

        # Calculate the Hamming distance
        hamm = CalcHammingDist(qB[iter, :], rB)

        # Sort the Hamming distance
        ind = np.argsort(hamm)

        # Reorder the ground truth based on the sorted Hamming distance
        gnd = gnd[ind]

        # Get the top k ground truth
        tgnd = gnd[0:topk]

        # Calculate the sum of the top k ground truth
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        # Creates a sequence of numbers from 1 to tsum
        count = np.linspace(1, tsum, tsum)

        # Finds the indices of relevant images in tgnd
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0

        # Calculates the top k mAP
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap


# Same as above but with pr curve, faster but more memory
def CalcTopMapWithPR(qB, queryL, rB, retrievalL, topk):
    num_query = queryL.shape[0]
    num_gallery = retrievalL.shape[0]
    topkmap = 0
    prec = np.zeros((num_query, num_gallery))
    recall = np.zeros((num_query, num_gallery))
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        all_sim_num = np.sum(gnd)

        prec_sum = np.cumsum(gnd)
        return_images = np.arange(1, num_gallery + 1)

        prec[iter, :] = prec_sum / return_images
        recall[iter, :] = prec_sum / all_sim_num

        assert recall[iter, -1] == 1.0
        assert all_sim_num == prec_sum[-1]

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    topkmap = topkmap.round(3)
    index = np.argwhere(recall[:, -1] == 1.0)
    index = index.squeeze()
    prec = prec[index]
    recall = recall[index]
    cum_prec = np.mean(prec, 0)
    cum_recall = np.mean(recall, 0)

    return topkmap, cum_prec, cum_recall


# https://github.com/chrisbyd/DeepHash-pytorch/blob/master/validate.py
# Validates the model
def validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset):
    device = config["device"]
    # print("calculating test binary code......")
    tst_binary, tst_label = compute_result(test_loader, net, device=device)

    # print("calculating dataset binary code.......")
    trn_binary, trn_label = compute_result(dataset_loader, net, device=device)

    # If the PR curve path is not in the config
    if "pr_curve_path" not in config:
        mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(), config["topK"])
    # If the PR curve path is in the config
    else:
        # Calculate the top map with the PR curve
        mAP, cum_prec, cum_recall = CalcTopMapWithPR(tst_binary.numpy(), tst_label.numpy(),
                                                     trn_binary.numpy(), trn_label.numpy(),
                                                     config["topK"])
        # Calculates how many full groups of 100 items that are in the dataset
        index_range = num_dataset // 100

        # Creates a list of indicies that correspond to the last item in each group of 100 items
        index = [i * 100 - 1 for i in range(1, index_range + 1)]

        # Finds the maximum value in the index list
        max_index = max(index)

        # Calculates the number of items that are not in a full group of 100 items
        overflow = num_dataset - index_range * 100

        # Adds the remaining items to the index list
        index = index + [max_index + i for i in range(1, overflow + 1)]

        # Calculates the cumulative precision and recall
        c_prec = cum_prec[index]
        c_recall = cum_recall[index]

        # Creates a dictionary of the index, precision, and recall
        pr_data = {
            "index": index,
            "P": c_prec.tolist(),
            "R": c_recall.tolist()
        }
        # Saves the pr curve
        os.makedirs(os.path.dirname(config["pr_curve_path"]), exist_ok=True)
        with open(config["pr_curve_path"], 'w') as f:
            f.write(json.dumps(pr_data))
        print("pr curve save to ", config["pr_curve_path"])

    # If the mean average precision is greater than the best mean average precision
    if mAP > Best_mAP:
        # Sets the best mean average precision to the mean average precision
        Best_mAP = mAP
        # Save the current model
        if "save_path" in config:
            save_path = os.path.join(config["save_path"], f'{config["dataset"]}_{bit}bits_{mAP.round(3)}')
            os.makedirs(save_path, exist_ok=True)
            print("save in ", save_path)
            np.save(os.path.join(save_path, "tst_label.npy"), tst_label.numpy())
            np.save(os.path.join(save_path, "tst_binary.npy"), tst_binary.numpy())
            np.save(os.path.join(save_path, "trn_binary.npy"), trn_binary.numpy())
            np.save(os.path.join(save_path, "trn_label.npy"), trn_label.numpy())
            torch.save(net.state_dict(), os.path.join(save_path, "model.pt"))
    print(f"{config['info']} epoch:{epoch + 1} bit:{bit} dataset:{config['dataset']} MAP:{mAP} Best MAP: {Best_mAP}")
    print(config)
    return Best_mAP
