import os
import sys


import torchvision.transforms as transforms
import torchvision.datasets as datasets

config_name = 'resnet18_cifar100_prune'

class Config(object):
    log = './log/' + config_name  # Path to save log
    checkpoint_path = './checkpoints/' + config_name  # Path to store checkpoint model
    resume = './checkpoints/' + config_name + '/latest.pth' # load checkpoint model
    evaluate = None  # evaluate model path

    input_image_size=32
    network = "resnet18"
    pretrained = False
    num_classes = 100
    seed = 2019211353

    train_dataset = datasets.CIFAR100(
        './data/cifar100',
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
        './data/cifar100',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                 std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]),
        ]))
    
    milestones = [60,120,160]
    epochs = 200
    batch_size = 128
    accumulation_steps = 1
    lr = 0.1
    weight_decay = 5e-4
    momentum = 0.9
    num_workers = 2
    print_interval = 100
    
    use_prune = True
    prune_interval = 782
    prune_num = 178670
    over_prune_threshold = 20
    prune_fail_times = 3
    
        
