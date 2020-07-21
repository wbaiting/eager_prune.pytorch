# import os
import sys

# BASE_DIR = os.path.dirname(
#     os.path.dirname(os.path.dirname(os.path.dirname(
#         os.path.abspath(__file__)))))
# sys.path.append(BASE_DIR)
# from public.path import pretrained_models_path


def get_network(args):
    """ return given network
    """

    if args.net == 'lenet':
        from models.lenet import lenet
        net = lenet()
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121(pretrained=args.pretrained, num_classes=args.num_classes)
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161(pretrained=args.pretrained, num_classes=args.num_classes)
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169(pretrained=args.pretrained, num_classes=args.num_classes)
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201(pretrained=args.pretrained, num_classes=args.num_classes)
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        if 'cifar' in args.dataset or 'mnist' in args.dataset:
            first_conv_3_3 = True
        else:
            first_conv_3_3 = False
        net = resnet18(pretrained=args.pretrained, num_classes=args.num_classes, first_conv_3_3=first_conv_3_3)
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        if 'cifar' in args.dataset or 'mnist' in args.dataset:
            first_conv_3_3 = True
        else:
            first_conv_3_3 = False
        net = resnet34(pretrained=args.pretrained, num_classes=args.num_classes, first_conv_3_3=first_conv_3_3)
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        if 'cifar' in args.dataset or 'mnist' in args.dataset:
            first_conv_3_3 = True
        else:
            first_conv_3_3 = False
        net = resnet50(pretrained=args.pretrained, num_classes=args.num_classes, first_conv_3_3=first_conv_3_3)
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        if 'cifar' in args.dataset or 'mnist' in args.dataset:
            first_conv_3_3 = True
        else:
            first_conv_3_3 = False
        net = resnet101(pretrained=args.pretrained, num_classes=args.num_classes, first_conv_3_3=first_conv_3_3)
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        if 'cifar' in args.dataset or 'mnist' in args.dataset:
            first_conv_3_3 = True
        else:
            first_conv_3_3 = False
        net = resnet152(pretrained=args.pretrained, num_classes=args.num_classes, first_conv_3_3=first_conv_3_3)
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    return net
