import random
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

DATA_ROOT = r'C:\Users\Sandra\PycharmProjects\learning_to_hash\datasets\cifar-10-batches-py'
BATCH_SIZE = 64
NUM_WORKERS = 8
CODE_SIZE = 8  # bits


class CIFAR10PairDataset(Dataset):
    def __init__(self, data_root: str, transform=None, train=True):
        super().__init__()
        self.dataset = CIFAR10(root=data_root, train=train, transform=transform, download=True)
        self.size = len(self.dataset)

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        # return image pair
        x_img, x_target = self.dataset[item]

        pair_idx = item
        while pair_idx == item:
            pair_idx = random.randint(0, self.size - 1)

        y_img, y_target = self.dataset[pair_idx]

        # similarity label (0 for same class, 1 for different class)
        target_equals = 0 if x_target == y_target else 1

        return x_img, x_target, y_img, y_target, target_equals

def print_sample_pairs(dataloader):
    for x_imgs, x_targets, y_imgs, y_targets, target_equals in dataloader:
        print("Sample pairs:")
        for i in range(3):
            print(f"Pair {i + 1}:")
            print(f"  x_target: {x_targets[i]}, y_target: {y_targets[i]}, target_equals: {target_equals[i]}")
        break

if __name__ == "__main__":
    cifar_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_pair_dataset = CIFAR10PairDataset(data_root=DATA_ROOT, train=True, transform=cifar_transform)
    test_pair_dataset = CIFAR10PairDataset(data_root=DATA_ROOT, train=False, transform=cifar_transform)

train_dataloader = DataLoader(
    train_pair_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=NUM_WORKERS
)
test_dataloader = DataLoader(
    test_pair_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=NUM_WORKERS
)

print(f'Train set size: {len(train_pair_dataset)}')
print(f'Test set size: {len(test_pair_dataset)}')

print_sample_pairs(train_dataloader)

class MNISTPairDataset(Dataset):
    def __init__(self, data_root: str, transform=None, train=True):
        super().__init__()
        self.dataset = MNIST(root=data_root, train=train, transform=transform, download=True)
        self.size = len(self.dataset)

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        # return image pair
        x_img, x_target = self.dataset[item]
        pair_idx = item
        # choose a different index
        while pair_idx == item:
            pair_idx = random.randint(0, self.size - 1)

        y_img, y_target = self.dataset[pair_idx]
        target_equals = 0 if x_target == y_target else 1
        return x_img, x_target, y_img, y_target, target_equals

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307, ), std=(0.3081, )),
])

train_pair_dataset = MNISTPairDataset(data_root=DATA_ROOT, train=True, transform=mnist_transform)
print(f'Train set size: {len(train_pair_dataset)}')
test_pair_dataset = MNISTPairDataset(data_root=DATA_ROOT, train=False, transform=mnist_transform)
print(f'Test set size: {len(test_pair_dataset)}')

train_dataloader2 = DataLoader(
    train_pair_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=NUM_WORKERS)
test_dataloader2 = DataLoader(
    test_pair_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=NUM_WORKERS)



