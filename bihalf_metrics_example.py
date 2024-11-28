import torch
from models.deep.deep_tools.tools import cifar_dataset
from models.deep.unsupervised_image_bit import BiHalfModelUnsupervised
from metrics.calc_metrics import calculate_metrics
from torchvision import transforms

"""
YOU SHOULD RUN THIS SCRIPT FROM THE ROOT DIRECTORY OF THE PROJECT
MAKE SURE THAT YOU HAVE A MODEL SAVED
IN ORDER TO CHANGE THE AMOUNT OF BITS THE MODEL USE, YOU WILL HAVE TO CHANGE THE MODEL BEFORE TRAINING OTHERWISE YOU WILL GET AN ERROR
"""

if __name__ == '__main__':
    config = {
            "resize_size": 256,
            "crop_size": 224,
            "batch_size": 8,
            # "dataset": "cifar10",
            "dataset": "cifar10-1",
            # "dataset": "cifar10-2",
            # "device":torch.device("cpu"),
            "device": torch.device("cuda:0"),
            "max_images": 16666, # 59000 is default amount of images
        }
        
    # Initialize the AlexNet model with x bits
    bihalf_cifar10 = BiHalfModelUnsupervised(config["batch_size"]).to(config["device"])

    # Load the model from the saved state
    bihalf_cifar10.load_state_dict(torch.load("save\BiHalf\cifar10-1_8bits_0.363\model.pt", map_location=config["device"]))

    # Get the data loaders for the CIFAR-10 dataset
    train_loader, test_loader, db_loader, _, _, _ = cifar_dataset(config)

    images = []
    labels = []
    num_loaded_images = 0
    # Get the images
    for batch in db_loader:
        batch_images, batch_labels, _ = batch
        
        images.append(batch_images)
        labels.append(batch_labels)
        
        num_loaded_images += batch_images.size(0)
        if num_loaded_images >= config["max_images"]:
            break
        
    # Concatenate the images and labels
    images = torch.cat(images)
    labels = torch.cat(labels)

    # Get query image, db_loader is 59000 with cifar10-1
    query_image = images[0].to(config["device"])

    # Find hash code for the query image, uses CUDA
    hash_code = bihalf_cifar10.query_with_cuda(query_image)
    
    # Print the hash code
    print(hash_code)
    
    # Calculate the metrics
    calculate_metrics(bihalf_cifar10.query_with_cuda, images, True)