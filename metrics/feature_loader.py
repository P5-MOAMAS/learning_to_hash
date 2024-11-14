import pickle
import sys
import psutil
import torch
import numpy as np
from metrics.hash_lookup import pre_gen_hash_codes


class FeatureLoader:
    def __init__(self, dataset_name: str):
        super().__init__()

        self.training = None
        self.test = None
        self.validation = None

        self.dataset_name = dataset_name

        match dataset_name:
            case "cifar-10":
                self.__init_cifar__()
            case _:
                print("Unrecognized feature set!")
                sys.exit(1)


    def __init_cifar__(self):
        # Get all features, from all images in cifar
        image_features = []
        image_labels = []
        for i in range(1, 6):
            name = "features/cifar-10-" + str(i) + "-features"

            with open(name, 'rb') as f:
                data = torch.load(f, weights_only=False)

            data = [np.array(i) for i in data]

            if data is None:
                print("No data was retrieved from '" + name + "'")
                sys.exit(1)
            image_features += data
        # Get all labels in cifar
        fp = "cifar-10-batches-py/"
        fileNames = []
        for i in range(1, 6):
            fileNames.append(fp + "data_batch_" + str(i))

        for name in fileNames:
            labels = unpickle(name)[b'labels']
            image_labels += labels

        if (len(image_features) != len(image_labels)):
            raise Exception("Not as many labels as there are images")

        idx_feature_label = []
        for id, feature_label in enumerate(zip(image_features, image_labels)):
            feature, label = feature_label
            idx_feature_label.append((id, feature, label))


        # Splitting dataset into 70% training 15 % testing and validation
        training_len = round(len(idx_feature_label) * 0.7)
        valid_test_len = (len(idx_feature_label) - training_len) // 2
        test_slice = training_len + valid_test_len

        self.training = idx_feature_label[0:training_len]
        self.test = idx_feature_label[training_len:test_slice]
        self.validation = idx_feature_label[test_slice:valid_test_len + test_slice]


def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


if __name__ == '__main__':
    def gen_key(some):
        return 0

    process = psutil.Process()
    mem_before = process.memory_info().rss

    fl = FeatureLoader("cifar-10")
    if fl.training == None or fl.validation == None or fl.test == None:
        raise Exception("One of the sets were null")

    print("Training:", len(fl.training), sep=" ")
    print("Validation:", len(fl.validation), sep=" ")
    print("Test:", len(fl.test), sep=" ")

    d = pre_gen_hash_codes(gen_key, fl.training) # Incorrect function declaration, works with this

    mem_after = process.memory_info().rss
    print("Before: " + str(mem_before))
    print("After: " + str(mem_after))
    print("Delta: " + str(mem_after - mem_before))

