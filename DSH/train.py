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
from model import LiuDSH

# hyper-parameters
DATA_ROOT = 'data_out'
LR_INIT = 3e-4
BATCH_SIZE = 128
NUM_WORKERS = 8
MARGIN = 5
ALPHA = 0.01

# check for cuda availability and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float)

def setup_data(dataset_name: str, data_root:str, train=True):
    """
       Configures a specified dataset with transformation.
       Supported datasets: MNIST, CIFAR-10, ImageNet.

       # Returns:
            dataset, size, channels and classes
       """

    if dataset_name == 'mnist':
        dataset = MNIST(root=data_root, train=train,
                        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.081,))]),
                        download=True)
        size, channels, classes = 28, 1, 10
    elif dataset_name == 'cifar':
        dataset = CIFAR10(root=data_root, train=train,
                          transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                          download=True)
        size, channels, classes = 32, 3, 10
    elif dataset_name == 'imagenet':
        dataset = ImageNet(root=data_root, train=train,
                           transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))]),
                           download=True)
        size, channels, classes = 256, 3, 1000
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')

    return dataset, size, channels, classes

class PairDataset(Dataset):
    """
        Dataset class that generates pairs of images and labels.
        For each sample, it randomly selects a pair with a similar or different label.

        # Returns:
            x_img, y_img (torch.tensor): the first and second image in the pair (randomly selected)
            x_target, y_target (torch.tensor): the first and second label in the pair
            target_equals (int): Binary indicator of similarity (0 if x_target == y_target, meaning the pair is similar, else 1 for dissimilar)
        """
    def __init__(self, data_root: str, dataset_name: str, train=True):
        super().__init__()

        # Load the dataset and retrieve metadata
        self.dataset, self.size, self.channels, self.classes = setup_data(dataset_name, data_root, train)

    def __len__(self):
        # Returns the total number of samples in the dataset
        return len(self.dataset)

    def __getitem__(self, item):
        # Fetch the primary image and its label
        x_img, x_target = self.dataset[item]

        pair_idx = item
        # Randomly select a different index for the second image in the pair
        while pair_idx == item:
            pair_idx = random.randint(0, self.size - 1)
        # Fetch the second image and its label
        y_img, y_target = self.dataset[pair_idx]
        # Set target_equals to 0 if labels match (similar), or 1 if they differ (dissimilar)
        target_equals = 0 if x_target == y_target else 1
        return x_img, x_target, y_img, y_target, target_equals,

class Trainer:
    def __init__(self):
        self.global_step = 0
        self.global_epoch = 0
        self.total_epochs = epochs
        self.input_shape = (dataset.channels, dataset.size, dataset.size)
        self.writer = SummaryWriter()
        self.writer.add_graph(model, self.generate_dummy_input(), verbose=True)

        # Define the loss functions (mean squared error for pairwise distance, L1 for regularization)
        self.mse_loss = nn.MSELoss(reduction='none')
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.optimizer = optim.Adam(model.parameters(), lr=LR_INIT)

    def __del__(self):
        # Ensure that the writer closes properly
        self.writer.close()

    def generate_dummy_input(self):
        # Generate a dummy input tensor for the model
        return torch.randn(1, *self.input_shape)

    def run_step(self, model, x_imgs, y_imgs, target_equals, train: bool):
        x_out = model(x_imgs) # Output for the first image (x_imgs)
        y_out = model(y_imgs) # Output for the second image (y_imgs)

        # squared_loss: Mean squared error between the model outputs for image pairs
        squared_loss = torch.mean(self.mse_loss(x_out, y_out), dim=1)
        # T1: 0.5 * (1 - y) * dist(x1, x2)
        # Compute the loss for similar pairs (target_equals = 0)
        positive_pair_loss = (0.5 * (1 - target_equals) * squared_loss)
        mean_positive_pair_loss = torch.mean(positive_pair_loss)

        # T2: 0.5 * y * max(margin - dist(x1, x2), 0)
        # Calculate the margin loss for dissimilar pairs (target_equals = 1)
        zeros = torch.zeros_like(squared_loss).to(device)
        margin = MARGIN * torch.ones_like(squared_loss).to(device)
        negative_pair_loss = 0.5 * target_equals * torch.max(zeros, margin - squared_loss)
        mean_negative_pair_loss = torch.mean(negative_pair_loss)

        # T3: alpha(dst_l1(abs(x1), 1)) + dist_l1(abs(x2), 1)))
        # Apply regularization on the absolute values of the outputs to push them towards +1 or -1
        mean_value_regularization = ALPHA * (
                self.l1_loss(torch.abs(x_out), torch.ones_like(x_out)) +
                self.l1_loss(torch.abs(y_out), torch.ones_like(y_out)))

        # Combine the positive pair loss, negative pair loss, and the regularization term
        self.loss = mean_positive_pair_loss + mean_negative_pair_loss + mean_value_regularization

        print(f'epoch: {self.global_epoch:02d}\t'
              f'step: {self.global_step:06d}\t'
              f'loss: {self.loss.item():04f}\t'
              f'positive_loss: {mean_positive_pair_loss.item():04f}\t'
              f'negative_loss: {mean_negative_pair_loss.item():04f}\t'
              f'regularize_loss: {mean_value_regularization:04f}')

        # log them to tensorboard
        self.writer.add_scalar('loss', self.loss.item(), self.global_step)
        self.writer.add_scalar('positive_pair_loss', mean_positive_pair_loss.item(), self.global_step)
        self.writer.add_scalar('negative_pair_loss', mean_negative_pair_loss.item(), self.global_step)
        self.writer.add_scalar('regularizer_loss', mean_value_regularization.item(), self.global_step)

        #  Backpropagation and optimization (if training)
        if train:
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()

        return x_out, y_out

    def train(self):
        for _ in range(self.total_epochs):
            # For each batch in the training dataset (train_dataloader)
            for x_imgs, x_targets, y_imgs, y_targets, target_equals in self.train_dataloader:
                target_equals = target_equals.type(torch.float)
                # Run a training step, updating the model parameters
                self.run_step(self.model, x_imgs, y_imgs, target_equals, train=True)
                self.global_step += 1

            # accumulate tensors for embeddings visualization
            test_imgs = []
            test_targets = []
            hash_embeddings = []
            embeddings = []

            #  Iterate over the test dataset for evaluation/visualization
            for test_x_imgs, test_x_targets, test_y_imgs, test_y_targets, test_target_equals in self.test_dataloader:
                test_target_equals = test_target_equals.type(torch.float)
                with torch.no_grad():
                    # Get embeddings for both images in the test set
                    x_embeddings, y_embeddings = self.run_step(
                        self.model, test_x_imgs, test_y_imgs, test_target_equals, train=False)

                if x_embeddings is None or y_embeddings is None:
                    print("Warning: Model output is None for the pair!")
                    continue  # Skip this batch if embeddings are invalid

                # Show all images that consist the pairs
                test_imgs.extend([test_x_imgs.cpu()[:5], test_y_imgs.cpu()[:5]])
                test_targets.extend([test_x_targets.cpu()[:5], test_y_targets.cpu()[:5]])

                # embedding1: hamming space embedding
                x_hash = torch.round(x_embeddings.cpu()[:5].clamp(-1, 1) * 0.5 + 0.5)
                y_hash = torch.round(y_embeddings.cpu()[:5].clamp(-1, 1) * 0.5 + 0.5)
                hash_embeddings.extend([x_hash, y_hash])

                # emgedding2: raw embedding
                embeddings.extend([x_embeddings.cpu(), y_embeddings.cpu()])

                self.global_step += 1

            # Log the raw embedding distribution to TensorBoard
            self.writer.add_histogram(
                'embedding_distribution',
                torch.cat(embeddings).cpu().numpy(),
                global_step=self.global_step)

            # Draw embeddings for a single batch - very nice for visualizing clusters
            self.writer.add_embedding(
                torch.cat(hash_embeddings),
                metadata=torch.cat(test_targets),
                label_img=torch.cat(test_imgs),
                global_step=self.global_step)

            # Tensor format
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
            print(result_text)

            self.global_epoch += 1

def train_model(dataset_name: str, code_size: int, epochs: int):
    """
          Handles loading the datasets, initialising its parameters (channels, size, classes) and passing those to training the model

          Args:
                dataset_name: The name of the dataset (mnist, cifar or imagenet)
                code_size: size of binary code  e.g. 8,16,32,64
                epochs: number of epochs
          """
    # Setup dataset and dataloaders
    train_pair_dataset = PairDataset(data_root=DATA_ROOT, dataset_name=dataset_name, train=True)
    print(f'Train set size: {len(train_pair_dataset)}')
    test_pair_dataset = PairDataset(data_root=DATA_ROOT, dataset_name=dataset_name, train=False)
    print(f'Test set size: {len(test_pair_dataset)}')

    channels = train_pair_dataset.channels
    size = train_pair_dataset.size
    classes = train_pair_dataset.classes

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

    print(f"Test dataloader has {len(test_dataloader)} batches")

    # Initialize model
    model = LiuDSH(code_size=code_size, channels=channels, size=size, num_classes=classes).to(device)

    # Create trainer and start training
    trainer = Trainer(model, train_dataloader, test_dataloader, train_pair_dataset, code_size, epochs)
    trainer.train()

    # Save the trained model
    torch.save(model.state_dict(), f'{dataset_name}_hash_model.pth')

if __name__ == '__main__':
    dataset = 'mnist' # Choose between mnist, cifar or imagenet
    code_size = 16 # Choose e.g. (8, 16, 32, 64)
    epochs = 5
    train_model(dataset_name=dataset, code_size=code_size, epochs=epochs)

    #For loading the model:
    #model.load_state_dict(torch.load('hash_model.pth', weights_only=True))
    #model.eval() #or train
    #hash_code = model.query_image(sample_image)
    #print("Hash Code:", hash_code)

