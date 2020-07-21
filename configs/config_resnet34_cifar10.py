import os
import sys


import torchvision.transforms as transforms
import torchvision.datasets as datasets

config_name = 'resnet34_cifar10'

class Config(object):
    log = './log/' + config_name  # Path to save log
    checkpoint_path = './checkpoints/' + config_name  # Path to store checkpoint model
    resume = './checkpoints/' + config_name + '/latest.pth' # load checkpoint model
    evaluate = None  # evaluate model path

    input_image_size=32
    network = "resnet34"
    pretrained = False
    num_classes = 10
    seed = 2019211353

    train_dataset = datasets.CIFAR10(
        './data/cifar10',
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124],
                                 std=[0.24703233, 0.24348505, 0.26158768]),
        ]))
    val_dataset = datasets.CIFAR10(
        './data/cifar10',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124],
                                 std=[0.24703233, 0.24348505, 0.26158768]),
        ]))
    
    milestones = [75,135,200]
    epochs = 256
    batch_size = 128
    accumulation_steps = 1
    lr = 0.1
    weight_decay = 1e-4
    momentum = 0.9
    num_workers = 8
    print_interval = 100
    
    use_prune = True
    prune_interval = 1000
    prune_num = 340282
    over_prune_threshold = 20
    prune_fail_times = 3
    
        
