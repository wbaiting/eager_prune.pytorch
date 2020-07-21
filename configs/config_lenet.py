import os
import sys


import torchvision.transforms as transforms
import torchvision.datasets as datasets

config_name = 'lenet_mnist'

class Config(object):
    log = './log/' + config_name  # Path to save log
    checkpoint_path = './checkpoints/' + config_name  # Path to store checkpoint model
    resume = './checkpoints/' + config_name + '/latest.pth' # load checkpoint model
    evaluate = None  # evaluate model path

    input_image_size=32
    network = "lenet5"
    pretrained = False
    num_classes = 10
    seed = 2019211353

    train_dataset = datasets.MNIST(
        './data/mnist',
        download=True,
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]))
    val_dataset = datasets.MNIST(
        './data/mnist',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]))
    
    milestones = [6,9]
    epochs = 11
    batch_size = 64
    accumulation_steps = 1
    lr = 0.01
    weight_decay = 1e-4
    momentum = 0.9
    num_workers = 8
    print_interval = 100
    
    use_prune = True
    prune_interval = 200
    prune_num = 808
    over_prune_threshold = 10
    prune_fail_times = 3
    
        
