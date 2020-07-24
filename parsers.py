import argparse


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def get_list(s):
    return [int(t) for t in s.strip().split(',')]

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--net',
                        type=str,
                        default='resnet50',
                        help='name of net')
    parser.add_argument('--gpus',
                        type=get_list,
                        default=(0,),
                        help='gpuid used in the training, e.g. --gpus=0,1,2')
    parser.add_argument('--lr',
                        type=float,
                        default=0.1,
                        help='learning rate')
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        help='momentum')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=5e-4,
                        help='weight decay')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.2,
                        help='gamma of lr_schedule')
    parser.add_argument('--warmup_epoch',
                        type=int,
                        default=1,
                        help='warmup epoch')
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='num of training epochs')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='batch size')
    parser.add_argument('--milestones',
                        type=list,
                        default=[30, 60, 90],
                        help='optimizer milestones')
    parser.add_argument('--pretrained',
                        type=boolean_string,
                        default=False,
                        help='load pretrained model params or not')
    parser.add_argument('--num_classes',
                        type=int,
                        default=1000,
                        help='model classification num')
    parser.add_argument('--input_image_size',
                        type=int,
                        default=224,
                        help='input image size')
    parser.add_argument('--num_workers',
                        type=int,
                        default=8,
                        help='number of worker to load data')
    parser.add_argument('--resume',
                        type=str,
                        default=False,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkpoints',
                        type=str,
                        default=None,
                        help='path for saving trained models')
    parser.add_argument('--log',
                        type=str,
                        default=None,
                        help='path to save log')
    parser.add_argument('--evaluate',
                        type=boolean_string,
                        default=False,
                        help='path for evaluate model')
    parser.add_argument('--seed', type=int, default=2019211353, help='seed')
    parser.add_argument('--print_interval',
                        type=bool,
                        default=100,
                        help='print interval')    
    parser.add_argument('--use_prune',
                       type=boolean_string,
                       default=True,
                       help='use eager_pruning or not')
    parser.add_argument('--prune_interval',
                       type=int,
                       default=1000,
                       help='prune_interval')
    parser.add_argument('--prune_num',
                       type=int,
                       default=888,
                       help='prune_num')
    parser.add_argument('--over_prune_threshold',
                       type=int,
                       default=20,
                       help='over_prune_threshold')
    parser.add_argument('--prune_fail_times',
                       type=int,
                       default=3,
                       help='prune_fail_times')
    parser.add_argument('--beishu',
                       type=float,
                       default=1.0,
                       help='beishu of max_loss')
    parser.add_argument('--force_flag',
                       type=boolean_string,
                       default=False,
                       help='force flag')
    parser.add_argument('--max_prune_rate',
                       type=float,
                       default=0.8,
                       help='max prune rate')
    parser.add_argument('--min_prune_rate',
                       type=float,
                       default=0.5,
                       help='min prune_rate')
    parser.add_argument('--check_beishu',
                       type=float,
                       default=0.4,
                       help='check beishu')
    parser.add_argument('--max_beishu',
                        type=float,
                        default=0.75,
                        help='max prune rate beishu')
    parser.add_argument('--dataset',
                        type=str,
                        default=None,
                        help='name of dataset')
    parser.add_argument('--config_path',
                        type=str,
                        default=None,
                        help='path to config')

    return parser.parse_args()

