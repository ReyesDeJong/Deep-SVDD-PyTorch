from .mnist import MNIST_Dataset
from .cifar10 import CIFAR10_Dataset
from .hits_dataset import HitsDataset


def load_dataset(dataset_name, data_path, normal_class):
    """Loads the dataset."""

    implemented_datasets = ('hits', 'mnist', 'cifar10')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'hits':
        dataset = HitsDataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path, normal_class=normal_class)

    return dataset
