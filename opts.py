import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument(
    'data',
    metavar='DIR',
    default='/data-sets/imagenet-eureka/imagenet256/',
    help='path to dataset')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                    metavar='W', help='weight decay (default: 1e-2)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument(
    '--multiprocessing-distributed',
    action='store_true',
    help='Use multi-processing distributed training to launch '
    'N processes per node, which has N GPUs. This is the '
    'fastest way to use PyTorch for either single node or '
    'multi node data parallel training')
parser.add_argument(
    '--optimizer',
    type=str,
    choices=[
        'sgd',
        'adam',
        'adamw'],
    default='adamw')
parser.add_argument('--use-mixup', action='store_true')
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--output-dir', type=str, default='output-exp/')
parser.add_argument(
    '--scheduler',
    type=str,
    choices=[
        'multistep',
        'cosine'],
    default='multistep')
parser.add_argument('--warmup', type=int, default=0)
parser.add_argument('--stem-type', type=str, default='basic')
parser.add_argument('--resume-epoch', action='store_true',
                    help='if enabled, will resume the epoch')

# Experts
parser.add_argument('--num-experts', type=int, default=4)
parser.add_argument('--use-only-first', action='store_true')
parser.add_argument(
    '--expansion-stage',
    action='store_true',
    help='If enabled it will replicate the weights from expert 0 to the rest.')

# Binarization
parser.add_argument(
    '--binary-activations',
    action='store_true',
    help='Binarize the activations')
parser.add_argument(
    '--binary-weights',
    action='store_true',
    help='Binarize the weights')

# Network structure
parser.add_argument('--structure', nargs='+', type=int, default=[2, 2, 2, 2])
parser.add_argument('--num-groups', nargs='+', type=int, default=[1, 1, 1, 1])
parser.add_argument('--expansion', nargs='+', type=float, default=[1, 1, 1, 1])
parser.add_argument('--downsample-ratio', type=int, default=4)
parser.add_argument(
    '--add-g-layer',
    action='store_true',
    help='if true, for g>1 a 1x1 layer will be added')
parser.add_argument('--use-se', action='store_true', help='Add a SE layer')

# distillation
parser.add_argument('--teacher', type=str, default='', help='teacher weights')
parser.add_argument(
    '--teacher-config',
    type=str,
    default='',
    help='path to Json containing the args to build the teacher model.')
parser.add_argument(
    '--att-transfer',
    default=False,
    action='store_true',
    help='Do attention transfer from real-valued network?')
parser.add_argument(
    '--att-transfer-weighting',
    default=1e+3,
    type=float,
    help='weighting of the att. transfer terms within the loss')
parser.add_argument(
    '--att-transfer-indicator',
    default=[
        0,
        1,
        1,
        1],
    type=int,
    nargs="+",
    help="Which stages to use for attention transfer")
parser.add_argument(
    '--lab-match',
    default=False,
    action='store_true',
    help='Match soft labels of the teacher network?')
parser.add_argument(
    '--lab-match-w',
    default=1e+3,
    type=float,
    help='weighting of the soft label matching loss term')
parser.add_argument(
    '--lab-match-T',
    default=1,
    type=float,
    help='Temperature to apply when computing the labels of the teacher')
