import os

import torch
from PIL import Image

def progressbar(iterable):
    length = len(iterable) - 1
    for i, item in enumerate(iterable):
        yield item
        progress = round(i/length * 30)
        print(f"|{("█" * progress)}{("-" * (30 - progress))}| {i + 1}/{length + 1}", end='\r', flush=True)
    print("", flush=True)


class ImageNetLoader:
    def __init__(self):
        self.paths = []
        self.target_types = {}

    """
    Internal function. Loading all locations of images on the disk from disk and stores them in the class
    """
    def __load_locations__(self):
        with open("data/ImageNet/ILSVRC/ImageSets/CLS-LOC/train_loc.txt", "r") as f:
            print("Loading paths of images...")
            for line in progressbar(f.readlines()):
                self.paths.append(line.split(" ")[0])


    """
    Internal function. Loading all targets from disk and stores them in the class
    """
    def __load_targets__(self):
        with open("data/ImageNet/LOC_synset_mapping.txt") as f:
            print("Loading target mappings...")
            for line in progressbar(f.readlines()):
                columns = line.split(" ")
                id = columns[0]
                types = []
                for i in range(1,len(columns)):
                    types.append(columns[i])
                self.target_types[id] = types


    """
    Loads all images within the range of start to end. Return a list of images in order
    start : the starting index to get images from
    end : the ending index to get images from
    """
    def load_images(self, start: int, end: int):
        if len(self.paths) == 0:
            self.__load_locations__()
        images = []
        print("Loading images...")
        for i in progressbar(range(start, end)):
            images.append(Image.open("data/ImageNet/ILSVRC/Data/CLS-LOC/train/" + self.paths[i] + ".JPEG").convert('RGB'))
        return images


    """
    Gets the full list of targets for each set of images. These are a list of string labels
    """
    def load_targets(self, start, end):
        if len(self.paths) == 0:
            self.__load_locations__()
        if len(self.target_types) == 0:
            self.__load_targets__()
        targets = []
        print("Loading targets...")
        for i in progressbar(range(start, end)):
            id = self.paths[i].split("/")[0]
            targets.append(self.target_types[id])
        return targets

    """
    Generates a single class number for each image in the dataset. 
    The original datasets classes are string labels
    """
    def load_single_numeric_targets(self):
        if len(self.paths) == 0:
            self.__load_locations__()
        if len(self.target_types) == 0:
            self.__load_targets__()

        target_types_num = {}
        next_id = 0
        # Give each class type a unique id
        for target_type_list in self.target_types:
            # Get the first class in the list, as it's the main class
            target_type = target_type_list[0]
            if target_type not in target_types_num:
                target_types_num[target_type] = next_id
                next_id += 1

        image_targets = []
        for i in range(0, 544546):
            id = self.paths[i].split("/")[0]
            image_targets.append(target_types_num[id])

        return image_targets


    """
    Extracts all labels and saves them to disk. One complete file containing all and one for each batch of features. 
    """
    def extract_class_labels(self):
        images = self.load_single_numeric_targets()
        print("Saving labels...")
        os.makedirs("labels/image-net-batch", exist_ok=True)
        torch.save(images, "labels/image-net-merged-labels")

        # Split labels into batches matching feature
        for batch in progressbar(range(1, 56)):
            batch_start = (batch - 1) * 10000
            batch_size = 10000 if batch != 55 else 4546
            file = "labels/image-net-batch/image-net-" + str(batch) + "-labels"
            labels = images[batch_start:batch_start + batch_size]
            torch.save(labels, file)


if __name__ == '__main__':
    loader = ImageNetLoader()
    loader.extract_class_labels()
