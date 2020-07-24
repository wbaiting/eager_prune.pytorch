import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
print(BASE_DIR)
from configs.path import ILSVRC2012_PATH, MNIST_PATH, CIFAR10_PATH, CIFAR100_PATH, IMAGENET_32_NOISE_PATH, IMAGENET_32_NOISE_PATH_2
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_dataset(args):
    """ return given network
    """

    if args.dataset == 'mnist':
        train_dataset = datasets.MNIST(
            MNIST_PATH,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]))
        val_dataset = datasets.MNIST(
            MNIST_PATH,
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]))
    elif args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(
            CIFAR10_PATH,
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124],
                                     std=[0.24703233, 0.24348505, 0.26158768]),
            ]))
        val_dataset = datasets.CIFAR10(
            CIFAR10_PATH,
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124],
                                     std=[0.24703233, 0.24348505, 0.26158768]),
            ]))
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(
            CIFAR100_PATH,
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                     std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]),
            ]))
        val_dataset = datasets.CIFAR100(
            CIFAR100_PATH,
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                     std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]),
            ]))
    elif args.dataset == 'imagenet':
        input_image_size = 224
        scale = 256 / 224
        train_dataset_path = os.path.join(ILSVRC2012_path, 'train')
        val_dataset_path = os.path.join(ILSVRC2012_path, 'val')
        train_dataset = datasets.ImageFolder(
            train_dataset_path,
            transforms.Compose([
                transforms.RandomResizedCrop(input_image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]))
        val_dataset = datasets.ImageFolder(
            val_dataset_path,
            transforms.Compose([
                transforms.Resize(int(input_image_size * scale)),
                transforms.CenterCrop(input_image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]))
    elif args.dataset == 'imagenet_32_noise':
        input_image_size = 224
        scale = 256 / 224
        train_dataset_path = os.path.join(IMAGENET_32_NOISE_PATH, 'train')
        val_dataset_path = os.path.join(IMAGENET_32_NOISE_PATH, 'val')
        train_dataset = datasets.ImageFolder(
            train_dataset_path,
            transforms.Compose([
                transforms.RandomResizedCrop(input_image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]))
        val_dataset = datasets.ImageFolder(
            val_dataset_path,
            transforms.Compose([
                transforms.Resize(int(input_image_size * scale)),
                transforms.CenterCrop(input_image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]))
    elif args.dataset == 'imagenet_32_noise_2':
        input_image_size = 224
        scale = 256 / 224
        train_dataset_path = os.path.join(IMAGENET_32_NOISE_PATH_2, 'train')
        val_dataset_path = os.path.join(IMAGENET_32_NOISE_PATH_2, 'val')
        train_dataset = datasets.ImageFolder(
            train_dataset_path,
            transforms.Compose([
                transforms.RandomResizedCrop(input_image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]))
        val_dataset = datasets.ImageFolder(
            val_dataset_path,
            transforms.Compose([
                transforms.Resize(int(input_image_size * scale)),
                transforms.CenterCrop(input_image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]))

    else:
        print('the dataset name you have entered is not supported yet')
        sys.exit()

    return train_dataset, val_dataset

