import numpy
import torch
import argparse


def parse_args(give_args=None):
    parser = argparse.ArgumentParser(description='UTLC')
    parser.add_argument('--cpu', action='store_true', help='use cpu only')
    parser.add_argument('--n_GPUs', type=int, default=1, help='number of GPUs')
    parser.add_argument('--apex', action='store_true')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dataset', default='cifarQ49',
                        choices=('cifar', 'imagenet_32', 'tiny', 'imagenet_64', 'cifarQ40', 'cifarQ49'))
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs to train')
    parser.add_argument('--stages', type=int, default=16)
    parser.add_argument('--mask_type', default='normal', choices=('uneven_1', 'uneven_2', 'normal', 'raster'))
    parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')
    parser.add_argument('--n_resgroups', type=int, default=9,
                        help='number of residual groups, 2 non local and N-2 locals')
    # Optimization specifications
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr_decay', type=int, default=25, help='learning rate decay per N epochs')
    parser.add_argument('--decay_type', type=str, default='step', help='learning rate decay type')
    parser.add_argument('--gamma', type=float, default=0.2, help='learning rate decay factor for step decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='ADAM beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='ADAM beta2')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='ADAM epsilon for numerical stability')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--load', type=str, default='none', choices=('none', 'best', 'last'))
    parser.add_argument('--keep', action='store_true')
    parser.add_argument('--position_encoding', action='store_true')
    parser.add_argument('--checkpoint', action='store_true')
    parser.add_argument('--no_writer', action='store_true')
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--print_every', type=int, default=40,
                        help='how many batches to wait before logging training status')
    args = parser.parse_args(give_args)
    if args.epochs == 0:
        args.epochs = 1e8
    for arg in vars(args):
        if vars(args)[arg] == 'True':
            vars(args)[arg] = True
        elif vars(args)[arg] == 'False':
            vars(args)[arg] = False
    if not torch.cuda.is_available():
        args.cpu = True
    if args.seed > 0:
        torch.manual_seed(args.seed)
        numpy.random.seed(args.seed)
        if not args.cpu:
            torch.cuda.manual_seed(args.seed)
    if args.position_encoding and args.mask_type.startswith('uneven'):
        raise ValueError('position encoding does not work with uneven masks')
    if args.position_encoding and args.stages == 4:
        raise ValueError('position encoding does not work with 4 stages')
    return args
