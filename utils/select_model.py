import os
import sys
import torch
from torchvision import models

BASE_DIR = os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from public.path import pretrained_models_path


model_urls = {
    'resnet18':
    '{}/resnet/resnet18-5c106cde.pth'.format(pretrained_models_path),
    
    'mobilenetv2':
    '{}/mobilenetv2/mobilenet_v2-b0353104.pth'.format(
        pretrained_models_path)
}


def get_network(args):
    """ return given network
    """

    if args.net == 'lenet':
        from models.lenet import lenet
        net = lenet()
    elif args.net == 'resnet18':
        net = models.resnet18(num_classes=args.num_classes)
        if 'cifar' in args.dataset or 'mnist' in args.dataset:
            net.conv1 = nn.Conv2d(3, net.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
            net.bn1 = nn.BatchNorm2d(net.inplanes)
            net.relu = nn.ReLU(inplace=True)
            net.maxpool = nn.Identity()
    elif args.net == 'mobilenetv2':
        net = models.mobilenet_v2(num_classes=args.num_classes)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.pretrained == True:
        net.load_state_dict(
                torch.load(model_urls[args.net]))
    return net
