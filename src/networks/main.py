from .mnist_LeNet import MNIST_LeNet, MNIST_LeNet_Autoencoder
from .cifar10_LeNet import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder
from .hits_LeNet import Hits_LeNet, Hits_LeNet_Autoencoder

def build_network(net_name):
    """Builds the neural network."""

    implemented_networks = ('hits_LeNet', 'mnist_LeNet', 'cifar10_LeNet')
    assert net_name in implemented_networks

    net = None

    if net_name == 'hits_LeNet':
        net = Hits_LeNet()

    if net_name == 'mnist_LeNet':
        net = MNIST_LeNet()

    if net_name == 'cifar10_LeNet':
        net = CIFAR10_LeNet()

    return net


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('hits_LeNet', 'mnist_LeNet', 'cifar10_LeNet')
    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'hits_LeNet':
        ae_net = Hits_LeNet_Autoencoder()

    if net_name == 'mnist_LeNet':
        ae_net = MNIST_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet':
        ae_net = CIFAR10_LeNet_Autoencoder()

    return ae_net
