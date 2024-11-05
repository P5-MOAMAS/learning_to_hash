"""
Liu et al., "Deep Supervised Hashing for Fast Image Retrieval"
"""
from collections import defaultdict
import random
import torch
from torch import nn
from torch import optim
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from model import LiuDSH

# hyper-parameters
DATA_ROOT = 'data_out'
LR_INIT = 3e-4
BATCH_SIZE = 128
EPOCH = 5
NUM_WORKERS = 8
CODE_SIZE = 8  # bits
MARGIN = 5
ALPHA = 0.01  # TODO: adjust
DATASET = 'cifar' # Choose between mnist, Cifar10, ImageNet
SIZE = 32 # Dimension of the picture e.g. 32 (32x32 pic)
CHANNELS = 3 # The amount of color channels

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_default_dtype(torch.float)

class PairDataset(Dataset):
    def __init__(self, data_root: str, transform=None, train=True):
        super().__init__()
        match DATASET:
            case 'mnist':
                self.dataset = MNIST(root=data_root, train=train, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.1307, ), std=(0.081, )),]), download=True)
                self.size = len(self.dataset)
            case 'cifar':
                self.dataset = CIFAR10(root=data_root, train=train, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.5, 0.5, 0.5),std=(0.5, 0.5, 0.5))]), download=True)
                self.size = len(self.dataset)
            case 'imagenet':
                self.dataset = ImageNet(root=data_root, train=train,transform=transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]), download=True)
                self.size = len(self.dataset)
            case _:
                raise ValueError(f'Unknown dataset {DATASET}')
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

model = LiuDSH(code_size=CODE_SIZE).to(device)

mse_loss = nn.MSELoss(reduction='none')
l1_loss = nn.L1Loss(reduction='mean')

optimizer = optim.Adam(model.parameters(), lr=LR_INIT)

class Trainer:
    def __init__(self):
        self.global_step = 0
        self.global_epoch = 0
        self.total_epochs = EPOCH

        self.input_shape = (channels, size, size)
        self.writer = SummaryWriter()
        self.writer.add_graph(model, self.generate_dummy_input(), verbose=True)

    def __del__(self):
        self.writer.close()

    def generate_dummy_input(self):
        return torch.randn(1, *self.input_shape)

    def run_step(self, model, x_imgs, y_imgs, target_equals, train: bool):
        # convert from double (float64) -> float32
        # TODO: dataset generates float64 by default?
        x_out = model(x_imgs)
        y_out = model(y_imgs)

        squared_loss = torch.mean(mse_loss(x_out, y_out), dim=1)

        # T1: 0.5 * (1 - y) * dist(x1, x2)
        positive_pair_loss = (0.5 * (1 - target_equals) * squared_loss)
        mean_positive_pair_loss = torch.mean(positive_pair_loss)

        # T2: 0.5 * y * max(margin - dist(x1, x2), 0)
        zeros = torch.zeros_like(squared_loss).to(device)
        margin = MARGIN * torch.ones_like(squared_loss).to(device)
        negative_pair_loss = 0.5 * target_equals * torch.max(zeros, margin - squared_loss)
        mean_negative_pair_loss = torch.mean(negative_pair_loss)

        # T3: alpha(dst_l1(abs(x1), 1)) + dist_l1(abs(x2), 1)))
        mean_value_regularization = ALPHA * (
                l1_loss(torch.abs(x_out), torch.ones_like(x_out)) +
                l1_loss(torch.abs(y_out), torch.ones_like(y_out)))

        loss = mean_positive_pair_loss + mean_negative_pair_loss + mean_value_regularization

        print(f'epoch: {self.global_epoch:02d}\t'
              f'step: {self.global_step:06d}\t'
              f'loss: {loss.item():04f}\t'
              f'positive_loss: {mean_positive_pair_loss.item():04f}\t'
              f'negative_loss: {mean_negative_pair_loss.item():04f}\t'
              f'regularize_loss: {mean_value_regularization:04f}')

        # log them to tensorboard
        self.writer.add_scalar('loss', loss.item(), self.global_step)
        self.writer.add_scalar('positive_pair_loss', mean_positive_pair_loss.item(), self.global_step)
        self.writer.add_scalar('negative_pair_loss', mean_negative_pair_loss.item(), self.global_step)
        self.writer.add_scalar('regularizer_loss', mean_value_regularization.item(), self.global_step)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return x_out, y_out

    def train(self):
        for _ in range(self.total_epochs):
            for x_imgs, x_targets, y_imgs, y_targets, target_equals in train_dataloader:
                target_equals = target_equals.type(torch.float)
                self.run_step(model, x_imgs, y_imgs, target_equals, train=True)
                self.global_step += 1

            # accumulate tensors for embeddings visualization
            test_imgs = []
            test_targets = []
            hash_embeddings = []
            embeddings = []

            for test_x_imgs, test_x_targets, test_y_imgs, test_y_targets, test_target_equals in test_dataloader:
                test_target_equals = test_target_equals.type(torch.float)
                with torch.no_grad():
                    x_embeddings, y_embeddings = self.run_step(
                        model, test_x_imgs, test_y_imgs, test_target_equals, train=False)

                # show all images that consist the pairs
                test_imgs.extend([test_x_imgs.cpu()[:5], test_y_imgs.cpu()[:5]])
                test_targets.extend([test_x_targets.cpu()[:5], test_y_targets.cpu()[:5]])

                # embedding1: hamming space embedding
                x_hash = torch.round(x_embeddings.cpu()[:5].clamp(-1, 1) * 0.5 + 0.5)
                y_hash = torch.round(y_embeddings.cpu()[:5].clamp(-1, 1) * 0.5 + 0.5)
                hash_embeddings.extend([x_hash, y_hash])

                # emgedding2: raw embedding
                embeddings.extend([x_embeddings.cpu(), y_embeddings.cpu()])

                self.global_step += 1

            self.writer.add_histogram(
                'embedding_distribution',
                torch.cat(embeddings).cpu().numpy(),
                global_step=self.global_step)

            # draw embeddings for a single batch - very nice for visualizing clusters
            self.writer.add_embedding(
                torch.cat(hash_embeddings),
                metadata=torch.cat(test_targets),
                label_img=torch.cat(test_imgs),
                global_step=self.global_step)

            # TODO: print text as hexadecimal strings
            hash_vals = torch.cat(hash_embeddings).numpy().astype(int)
            hash_vals = np.packbits(hash_vals, axis=-1).squeeze()  # to uint8
            targets = torch.cat(test_targets).numpy().astype(int)

            hashdict = defaultdict(list)
            for target_class, hash_value in zip(targets, hash_vals):
                hashdict[target_class].append(f'{hash_value:#04x}')  # ex) 15 -> 0x0f

            result_texts = []  # TODO: debug
            for target_class in sorted(hashdict.keys()):
                for hashval in hashdict[target_class]:
                    result_texts.append(f'class: {target_class:02d} - {hashval}')
                    self.writer.add_text(
                        f'e{self.global_epoch}_hashvals/{target_class:02d}',
                        hashval, global_step=self.global_step)

            result_text = '\n'.join(result_texts)
            print(result_text)  # TODO debug

            self.global_epoch += 1

if __name__ == '__main__':
    train_pair_dataset = PairDataset(data_root=DATA_ROOT, train=True)
    print(f'Train set size: {len(train_pair_dataset)}')
    test_pair_dataset = PairDataset(data_root=DATA_ROOT, train=False)
    print(f'Test set size: {len(test_pair_dataset)}')

    train_dataloader = DataLoader(
        train_pair_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS)
    test_dataloader = DataLoader(
        test_pair_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS)

    trainer = Trainer(SIZE,CHANNELS)
    trainer.train()
