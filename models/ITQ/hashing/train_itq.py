from itq_model import ITQ
from dataset import *
from evaluation import mean_average_precision, precision_recall

def load_dataset(dataset_name):
    """
    Function to load in the different datasets based on dataset_name.

    # Parameters:
        dataset_name: str
            The name of the dataset to load.

    # Returns:
        query_data: tuple[array, array]
            A tuple containing features and labels for the query set.
        database_data: tuple[array, array]
            A tuple containing features and labels for the database set.
    """
    match dataset_name:
        case 'mnist':
            root = './datasets/mnist-gist-512d.npz'
            query_data, train_data, database_data = load_mnist_gist(root)
        case 'cifar10_deep':
            root = './deep_features/cifar10_features'
            query_data, database_data = load_cifar10_deep(root)
        case 'cifar10_gist':
            root = './datasets/'
            query_data, database_data = load_cifar10_gist(root)
        case 'imagenet_deep':
            root = './imagenet_deep_features/'
            query_data, database_data = load_imagenet_deep(root)
        case _:
            raise ValueError(f'Unknown dataset: {dataset_name}')
    return query_data, database_data

def initialize_itq(encode_len):
    """
    Initialize the ITQ model with the given encode length.

    # Parameters:
        encode_len: int
            The length of the binary codes for ITQ encoding.

    # Returns:
        model: ITQ
            An initialized ITQ model instance.
    """
    model = ITQ(encode_len)
    return model

def fit_and_save_model(model, database, model_path='itq_model.pth'):
    """
    Fit the ITQ model and save it to a file.

    # Parameters:
        model: ITQ
            The ITQ model instance to be trained.
        database: array
            The database features used for training.
        model_path: str, optional
            The file path where the trained model will be saved.
    """
    model.fit(database)
    model.save_model(model_path)

def load_and_encode_model(model, query, database, model_path='itq_model.pth'):
    """
    Load the ITQ model from a file and encode the query and database.

    # Parameters:
        model: ITQ
            The ITQ model instance.
        query: array
            The query features to be encoded.
        database: array
            The database features to be encoded.
        model_path: str, optional
            The file path from which the trained model will be loaded.

    # Returns:
        query_b: array
            The binary codes for the query set.
        database_b: array
            The binary codes for the database set.
    """
    loaded_model = ITQ(model.encode_len)
    loaded_model.load_model(model_path)
    query_b = loaded_model.encode(query)
    database_b = loaded_model.encode(database)
    return query_b, database_b

def evaluate_model(query_b, query_label, database_b, database_label):
    """
    Evaluate the ITQ model by calculating mAP, precision, and recall.

    # Parameters:
        query_b: array
            The binary codes for the query set.
        query_label: array
            The labels for the query set.
        database_b: array
            The binary codes for the database set.
        database_label: array
            The labels for the database set.

    # Returns:
        mAP: float
            The mean Average Precision of the model.
        precision: array
            The precision values.
        recall: array
            The recall values.
    """
    mAP = mean_average_precision(query_b, query_label, database_b, database_label)
    precision, recall = precision_recall(query_b, query_label, database_b, database_label)
    return mAP, precision, recall

def print_metrics(mAP, precision, recall):
    """
    Print the evaluation metrics.

    # Parameters:
        mAP: float
            The mean Average Precision of the model.
        precision: array
            The precision values.
        recall: array
            The recall values.
    """
    print(f'mAP = {mAP}')
    print("Precision:")
    print(precision)
    print("Recall:")
    print(recall)

def train(dataset_name, encode_len):
    """
    Main function to load data, initialize the model, fit and save the model,
    load the model, encode the data, evaluate the model, and print the metrics.

    # Parameters:
        dataset_name: str
            The name of the dataset to use for training.
        encode_len: int
            The length of the binary codes for ITQ encoding.
    """
    query_data, database_data = load_dataset(dataset_name)
    query, query_label = query_data
    database, database_label = database_data

    print(f'Query shape: {query.shape}, Labels shape: {query_label.shape}')
    print(f'Database shape: {database.shape}, Labels shape: {database_label.shape}')

    model = initialize_itq(encode_len)
    fit_and_save_model(model, database)
    query_b, database_b = load_and_encode_model(model, query, database)
    mAP, precision, recall = evaluate_model(query_b, query_label, database_b, database_label)
    print_metrics(mAP, precision, recall)

if __name__ == '__main__':
    dataset_name = 'mnist'  # Change this as needed
    encode_len = 8          # Change this as needed (e.g., 8, 16, 32, 64)
    train(dataset_name, encode_len)
