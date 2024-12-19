import numpy as np
from PIL import Image


class NuswideMLoader:
    def __init__(self):
        self.data_root = "data/nus_wide_m/"
        self.data = []
        self.labels = []
        self.get_data()


    """
    Loads the Nuswide path and label file
    """
    def load_image_paths(self, path: str):
        data = []
        labels = []
        # Load the file
        image_list = open(path).readlines()
        for val in image_list:
            # Get the path from the first "word" in the list
            path = self.data_root + val.split()[0]
            # Labels are everything after the first "word"
            image_labels = np.array([int(label) for label in val.split()[1:]])
            data.append(path)
            labels.append(image_labels)

        return data, labels

    """
    Prepare the data paths and labels for train, database and test
    """
    def get_data(self):
        dataset_paths = [
            self.data_root + "train.txt",
            self.data_root + "database.txt",
            self.data_root + "test.txt"
        ]

        # Load All data paths and labels
        for path in dataset_paths:
            data, labels = self.load_image_paths(path)
            self.data.extend(data)
            self.labels.extend(labels)

    def __getitem__(self, index):
        path = self.data[index]
        img = Image.open(path).convert('RGB')
        return img

    def __len__(self):
        return len(self.labels)
