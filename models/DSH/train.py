"""
Liu et al., "Deep Supervised Hashing for Fast Image Retrieval"
"""
import time
from collections import defaultdict
import random
import torch
from torch import nn
from torch import optim
from torchvision.datasets import MNIST, ImageNet, CIFAR10
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from .model import LiuDSH
from tqdm import tqdm  # Add tqdm for progress bar
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
    def __init__(self, model, train_dataloader, test_dataloader, dataset: PairDataset, code_size: int, epochs: int):
        self.code_size = code_size
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.global_step = 0
        self.global_epoch = 0
        self.total_epochs = epochs
        self.input_shape = (dataset.channels, dataset.size, dataset.size)
        self.writer = SummaryWriter()
        self.writer.add_graph(model, self.generate_dummy_input(), verbose=True)

        self.mse_loss = nn.MSELoss(reduction='none')
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.optimizer = optim.Adam(model.parameters(), lr=LR_INIT)

        self.best_loss = float('inf')  # Initialize best loss to a high value
        self.best_epoch = -1  # To track which epoch had the best model
        self.best_model_path = "best_model.pth"  # File name for saving the best model

    def generate_dummy_input(self):
        return torch.randn(1, *self.input_shape, device=device)

    def __del__(self):
        self.writer.close()

    def run_step(self, model, x_imgs, y_imgs, target_equals, train: bool):
        x_out = model(x_imgs)
        y_out = model(y_imgs)

        squared_loss = torch.mean(self.mse_loss(x_out, y_out), dim=1)
        positive_pair_loss = (0.5 * (1 - target_equals) * squared_loss)
        mean_positive_pair_loss = torch.mean(positive_pair_loss)

        zeros = torch.zeros_like(squared_loss).to(device)
        margin = MARGIN * torch.ones_like(squared_loss).to(device)
        negative_pair_loss = 0.5 * target_equals * torch.max(zeros, margin - squared_loss)
        mean_negative_pair_loss = torch.mean(negative_pair_loss)

        mean_value_regularization = ALPHA * (
                self.l1_loss(torch.abs(x_out), torch.ones_like(x_out)) +
                self.l1_loss(torch.abs(y_out), torch.ones_like(y_out)))

        self.loss = mean_positive_pair_loss + mean_negative_pair_loss + mean_value_regularization

        # Log to tensorboard
        self.writer.add_scalar('loss', self.loss.item(), self.global_step)
        self.writer.add_scalar('positive_pair_loss', mean_positive_pair_loss.item(), self.global_step)
        self.writer.add_scalar('negative_pair_loss', mean_negative_pair_loss.item(), self.global_step)
        self.writer.add_scalar('regularizer_loss', mean_value_regularization.item(), self.global_step)

        if train:
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()

        return x_out, y_out

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        count = 0
        with torch.no_grad():
            for x_imgs, x_targets, y_imgs, y_targets, target_equals in self.test_dataloader:
                target_equals = target_equals.type(torch.float)
                _, _ = self.run_step(self.model, x_imgs, y_imgs, target_equals, train=False)
                total_loss += self.loss.item()
                count += 1
        avg_loss = total_loss / count
        self.writer.add_scalar('loss', avg_loss, self.global_epoch)
        return avg_loss

    def train(self):
        start_time = time.time()

        for epoch in range(self.total_epochs):
            self.global_epoch = epoch
            progress_bar = tqdm(
                total=len(self.train_dataloader),
                desc=f"Epoch {epoch + 1:02d}",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} batches",
                leave=False,
            )

            self.model.train()

            for x_imgs, x_targets, y_imgs, y_targets, target_equals in self.train_dataloader:
                target_equals = target_equals.type(torch.float)
                self.run_step(self.model, x_imgs, y_imgs, target_equals, train=True)
                self.global_step += 1

                # Update progress bar with current losses
                progress_bar.set_postfix({
                    'step': f'{self.global_step:06d}',
                    'loss': f'{self.loss.item():.04f}',
                })
                progress_bar.update(1)

            progress_bar.close()

            # Evaluate the model at the end of the epoch
            #avg_loss = self.evaluate()
            #print(f"Epoch {epoch + 1}/{self.total_epochs}, Validation Loss: {avg_loss:.4f}")

            self.best_model_path = f'best_model_{dataset}_{code_size}.pth'

            # Save the model if it has the best performance so far
            if self.loss < self.best_loss:
                self.best_loss = self.loss
                self.best_epoch = epoch + 1
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"New best model saved with loss {self.best_loss:.4f}")

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Training complete. Total training time: {total_time:.2f} seconds.")
        print(f"Training complete. Best model saved at {self.best_model_path} with loss {self.best_loss:.4f} from epoch {self.best_epoch} with time: {end_time:.2f }")


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

def get_image_hash(model_path: str, image: torch.Tensor, dataset_name: str, code_size: int):
    dataset, size, channels, classes = setup_data(dataset_name, DATA_ROOT)
    model = LiuDSH(code_size=code_size, channels=channels, size=size, num_classes=classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model.query_image(image)

if __name__ == '__main__':
    dataset = 'cifar' # Choose between mnist, cifar or imagenet
    code_size = 16 # Choose e.g. (8, 16, 32, 64)
    epochs = 50
    train_model(dataset_name=dataset, code_size=code_size, epochs=epochs)

    # Example of querying hash code for an image
    test_pair_dataset = PairDataset(DATA_ROOT, dataset, train=False)
    test_image, _ = test_pair_dataset[0][:2]
    hash_code = get_image_hash('best_model.pth', test_image, dataset, code_size)
    print("Hash Code for the test image:", hash_code)

    #For loading the model:
    #model.load_state_dict(torch.load('hash_model.pth', weights_only=True))
    #model.eval() #or train
    #hash_code = model.query_image(sample_image)
    #print("Hash Code:", hash_code)

