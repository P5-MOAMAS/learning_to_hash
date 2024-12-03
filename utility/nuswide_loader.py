import numpy as np
from PIL import Image


class NuswideMLoader:
    def __init__(self):
        self.data_root = "data/nus_wide_m/"
        self.data = []
        self.labels = []
        self.get_data()

    def load_image_paths(self, path: str):
        data = []
        labels = []
        image_list = open(path).readlines()
        print("Path: " + path + ", length: " + str(len(image_list)))
        for val in image_list:
            path = self.data_root + val.split()[0]
            # Labels are everything after the first "word"
            image_labels = np.array([int(label) for label in val.split()[1:]])
            data.append(path)
            labels.append(image_labels)

        return data, labels

    def get_data(self):
        dataset_paths = [
            self.data_root + "train.txt",
            self.data_root + "database.txt",
            self.data_root + "test.txt"
        ]

        for path in dataset_paths:
            data, labels = self.load_image_paths(path)
            self.data.extend(data)
            self.labels.extend(labels)

    def __getitem__(self, index):
        path = self.data[index]
        target = self.labels[index]
        img = Image.open(path).convert('RGB')
        return img, target

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    loader = NuswideMLoader()

    print(len(loader.data), len(loader.labels))
    print(loader.data[0])
