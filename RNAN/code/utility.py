import os
import time
import torch
import shutil
import datetime
import subprocess
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs

try:
    from torch.utils.tensorboard import SummaryWriter

    has_writer = False
except:
    has_writer = False

try:
    from apex.optimizers import FusedAdam as Adam
except ImportError:
    print('install apex for better performance')

from masks import *


class Timer:
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0
        return ret

    def reset(self):
        self.acc = 0


class Checkpoint:
    def __init__(self, args):
        if not args.test:
            try:
                subprocess.check_output('git diff-index --quiet HEAD --'.split())
            except:
                raise ValueError('Do not run an actual experiment with uncommitted changes')
        if args.no_writer:
            global has_writer
            has_writer = False
        self.args = args
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        self.dir = '../experiment/' + args.exp_name
        if os.path.exists(self.dir) and args.load == 'none':
            print('Warning, you should set resume to non zero if you want to continue training')
            user_input = input('should we start from scratch?(y/n)')
            if not user_input.startswith('y'):
                exit(0)
            else:
                shutil.rmtree(self.dir)
        if not os.path.exists(self.dir):
            args.load = 'none'

        def _make_dir(path):
            os.makedirs(path, exist_ok=True)

        _make_dir(self.dir)
        _make_dir(self.dir + '/model')
        _make_dir(self.dir + '/results')
        if has_writer:
            self.writer = SummaryWriter(self.dir + '/runs/' + now)
        else:
            self.writer = None
        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            if not args.test:
                f.write('\ngit-commit: ' + str(subprocess.check_output(['git', 'rev-parse', 'HEAD'])))

    def write_log(self, log, print_as_well=False):
        if print_as_well:
            print(log)
        self.log_file.write(log + '\n')
        self.log_file.flush()

    def add_scalar(self, *args, **kwargs):
        if self.writer is not None:
            self.writer.add_scalar(*args, **kwargs)

    def done(self):
        self.log_file.close()
        if self.writer is not None:
            self.writer.close()

    def get_state_dict(self, model, is_distributed):
        if model is None:
            return None
        if self.args.n_GPUs == 1 or not is_distributed:
            target = model
        else:
            target = model.module
        return target.state_dict()

    def save(self, trainer, is_best, keep=False, epoch=None):
        latest_path = os.path.join(self.dir, 'model', 'last.pt')
        if keep and os.path.exists(latest_path):
            shutil.copy(latest_path, os.path.join(self.dir, 'model', 'ckp_{}.pt'.format(epoch)))
        torch.save(dict(
            model=self.get_state_dict(trainer.model, True),
            optimizer=self.get_state_dict(trainer.optimizer, False),
            scheduler=self.get_state_dict(trainer.scheduler, False),
            best_loss=trainer.best_val_loss,
            best_bpsp=trainer.best_val_bpsp,
            epoch=trainer.current_epoch, trainer_counter=trainer.loader_train.counter
        ), latest_path)
        if is_best:
            shutil.copy(latest_path, os.path.join(self.dir, 'model', 'best.pt'))
        self.log_file.flush()


def trainable_parameters(model):
    return filter(lambda x: x.requires_grad, model.parameters())


def make_optimizer(args, my_model):
    trainable = trainable_parameters(my_model)
    kwargs = {'betas': (args.beta1, args.beta2), 'eps': args.epsilon, 'lr': args.lr, 'weight_decay': args.weight_decay}
    return Adam(trainable, **kwargs)


def make_scheduler(args, optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            optimizer,
            step_size=args.lr_decay,
            gamma=args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=args.gamma
        )
    else:
        raise ValueError('invalid scheduler')
    return scheduler


def count_parameters(model):
    return sum(p.numel() for p in trainable_parameters(model))


def generate_map(stages, size, args):
    if stages == 16:
        return generate_16_map(size, args.cpu, args.mask_type)
    if stages == 4:
        return generate_4_map(size, args.cpu)
    raise ValueError('invalid number of stages')


def generate_4_map(size, cpu):
    assert size % 2 == 0
    mask_map = {
        0: np.array([[0, 0],
                     [0, 0]]),
        1: np.array([[1, 0],
                     [0, 0]]),
        2: np.array([[1, 0],
                     [0, 1]]),
        3: np.array([[1, 1],
                     [0, 1]]),
        4: np.array([[1, 1],
                     [1, 1]])
    }
    mask_map = {k: torch.from_numpy(np.tile(v, (size // 2, size // 2)).astype(np.float32))
                for k, v in mask_map.items()}
    if not cpu:
        mask_map = {k: v.cuda() for k, v in mask_map.items()}
    return torch.stack([mask_map[i] for i in range(5)], dim=0)


def generate_16_map(size, cpu, mask_type):
    assert size % 4 == 0
    mask_map = total_mask[mask_type]

    mask_map = {i: torch.from_numpy(
        np.tile(mask_map[i], (size // mask_map.shape[1], size // mask_map.shape[1])).astype(np.float32)) for i in
        range(mask_map.shape[0])}
    if not cpu:
        mask_map = {k: v.cuda() for k, v in mask_map.items()}
    return torch.stack([mask_map[i] for i in range(17)], dim=0)


def get_size_of_file(path):  # in bytes
    return os.stat(path).st_size


def int_quantize(image, range=255):
    return np.clip((image * range + 0.5), 0, range).astype(np.uint8 if range == 255 else np.int16)


def get_patch_size(dataset):
    if dataset in ('cifar', 'imagenet32', 'cifarQ40', 'cifarQ49'):
        return 32
    if dataset in ('tiny', 'imagenet64'):
        return 64
    raise ValueError('invalid dataset name')


def place_model(model, args):
    if model is None:
        return model
    if not args.cpu:
        model = model.cuda()
        if args.n_GPUs > 1:
            model = nn.DataParallel(model, range(0, args.n_GPUs))
    return model


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    # TF ordering

    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))


def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def discretized_mix_logistic_loss(x, l):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Pytorch ordering

    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = 10
    logit_probs = l[:, :, :, :nr_mix * 3].contiguous().view(
        xs + [nr_mix])
    l = l[:, :, :, nr_mix * 3:].contiguous().view(
        xs + [nr_mix * 3])  # 3 for mean, scale, coef
    means = l[:, :, :, :, :nr_mix]
    # log_scales = torch.max(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)

    coeffs = torch.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels

    x = x.contiguous()
    x = x.unsqueeze(-1) + torch.zeros(xs +
                                      [nr_mix]).to(x)
    m2 = (means[:, :, :, 1, :] + coeffs[:, :, :, 0, :]
          * x[:, :, :, 0, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    m3 = (means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
          coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # now select the right output: left edge case, right edge case, normal
    # case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation
    # based on the assumption that the log-density is constant in the bin of
    # the observed sub-pixel value

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * torch.log(torch.clamp(
        cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (log_pdf_mid - np.log(127.5))
    inner_cond = (x > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + \
                (1. - inner_cond) * inner_inner_out
    cond = (x < -0.999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs = log_probs + log_prob_from_logits(logit_probs)
    return -(log_sum_exp(log_probs)).sum(dim=-1)


def none_zero_sum(tensor):
    tensor_sum = tensor.sum()
    if tensor_sum.item() == 0:
        return 1.0
    return tensor_sum
