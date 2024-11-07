from PIL import Image

from torchvision.transforms import transforms

from progressbar import progressbar


class ImageNetLoader:
    def __init__(self):
        self.paths = []
        self.target_types = {}

    def __load_locations__(self):
        with open("data/ImageNet/ILSVRC/ImageSets/CLS-LOC/train_loc.txt", "r") as f:
            print("Loading paths of images...")
            for line in progressbar(f.readlines()):
                self.paths.append(line.split(" ")[0])


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


    def load_images(self, start, end, trans = None):
        if len(self.paths) == 0:
            self.__load_locations__()
        images = []
        print("Loading images...")
        for i in progressbar(range(start, end)):
            image = Image.open("data/ImageNet/ILSVRC/Data/CLS-LOC/train/" + self.paths[i] + ".JPEG").convert('RGB')
            if trans is not None:
                image = trans(image)
            images.append(image)
        return images

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

if __name__ == '__main__':
    loader = ImageNetLoader()
    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    images = loader.load_images(0, 10, trans)

    print(len(images), images[0].shape)
