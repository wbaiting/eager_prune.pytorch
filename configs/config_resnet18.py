import os
import sys


ILSVRC2012_path = '/raid/data/image/data/'
import torchvision.transforms as transforms
import torchvision.datasets as datasets


config_name = 'resnet18_imagenet'

class Config(object):
    log = './log/' + config_name  # Path to save log
    checkpoint_path = './checkpoints/' + config_name  # Path to store checkpoint model
    resume = './checkpoints/' + config_name + '/latest.pth' # load checkpoint model
    evaluate = None  # evaluate model path
    train_dataset_path = os.path.join(ILSVRC2012_path, 'train')
    val_dataset_path = os.path.join(ILSVRC2012_path, 'val')

    network = "resnet18"
    pretrained = False
    num_classes = 1000
    seed = 2019211353
    input_image_size = 224
    scale = 256 / 224

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

    milestones = [30, 60, 90]
    epochs = 100
    batch_size = 256
    accumulation_steps = 1
    lr = 0.1
    weight_decay = 1e-4
    momentum = 0.9
    num_workers = 8
    print_interval = 100
    
    use_prune = True
    prune_interval = 5004
    prune_num = 178670
    over_prune_threshold = 20
    prune_fail_times = 3
