import os
import torch
import numpy as np
from tqdm import trange
from decimal import Decimal

import utility
from model.rnan import RNAN

try:
    from apex import amp
except ImportError:
    pass


class Trainer:
    def __init__(self, args, loader, checkpoint):
        self.args = args
        self.ckp = checkpoint
        self.loader_train = iter(loader.loader_train)
        self.loader_val = loader.loader_val
        self.model = RNAN(args)
        if not self.args.cpu:
            self.model = self.model.cuda()
        self.patch_size = utility.get_patch_size(args.dataset)
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)
        self.stages = args.stages
        self.mask_map = utility.generate_map(self.stages, self.patch_size, args)

        if self.stages == 16 and args.mask_type.startswith('uneven'):
            if args.mask_type == 'uneven_1':
                self.stage_weights = np.array([1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 16], dtype=np.float32)
            else:
                self.stage_weights = np.array([1, 1, 1, 1, 2, 2, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8], dtype=np.float32)
        else:
            self.stage_weights = np.ones(self.stages, dtype=np.float32)
        self.best_val_loss = float('inf')
        self.best_val_bpsp = float('inf')
        self.is_fast_test = args.test
        self.current_epoch = 0
        if self.args.load != 'none':
            self.load_all(self.args.load)
        try:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                                                        opt_level='O1' if args.apex else 'O0')
        except:
            pass
        self.model = utility.place_model(self.model, args)
        self.ckp.write_log('number of parameters in the enhancer: {}'.format(utility.count_parameters(self.model)))
        self.eps = np.finfo(np.float32).eps.item()

    def load_all(self, mode):
        kwargs = {'map_location': lambda storage, loc: storage} if self.args.cpu else {}
        state = torch.load(os.path.join(self.ckp.dir, 'model/{}.pt'.format(mode)), **kwargs)
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.best_val_loss = state['best_loss']
        self.best_val_bpsp = state['best_bpsp']
        self.current_epoch = state['epoch']

    def calc_loss(self, sr, hr, current_mask, next_mask):
        lp = utility.discretized_mix_logistic_loss(hr, sr[:, :120])
        effective_mask = (next_mask - current_mask).squeeze(1)  # B * H * W
        effective_sum = effective_mask.sum(dim=(1, 2), keepdim=True)  # B * 1 * 1
        policy_loss = (((lp * effective_mask) / effective_sum).sum(dim=(1, 2))).mean()
        return policy_loss

    def get_batch_stage_from_int(self, stage, B):
        batch_stage = torch.from_numpy(np.array([stage], dtype=np.int64)).expand(B)
        if not self.args.cpu:
            batch_stage = batch_stage.cuda()
        return batch_stage

    def generate_mask(self, B, H, W, stage):
        batch_stage = self.get_batch_stage_from_int(stage, B)
        mask = self.mask_map[stage].view(1, 1, H, W).expand(B, -1, -1, -1)
        next_mask = self.mask_map[stage + 1].view(1, 1, H, W).expand(B, -1, -1, -1)
        return mask, next_mask, batch_stage

    def get_stages_per_batch(self):
        return self.stages

    def step_stage(self, is_train, stage, B, H, W, orig_lr, hr, hidden_state, total_loss,
                   stages_per_batch, recurrent_loss, loss_per_stage):
        mask, next_mask, batch_stage = self.generate_mask(B, H, W, stage)
        lr = torch.cat([orig_lr, hr * mask, mask], dim=1)
        if is_train and stage == 0:
            self.optimizer.zero_grad()
        sr, hidden_state = self.model(lr, batch_stage, hidden_state)
        loss = self.calc_loss(sr, hr, mask, next_mask)
        total_loss += loss.item()
        if is_train:
            recurrent_loss = recurrent_loss + loss
            if stage == (stages_per_batch - 1):
                loss = recurrent_loss / stages_per_batch
                try:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                except:
                    loss.backward()
                self.optimizer.step()
        else:
            loss_per_stage[stage].append(loss.item())
        return hidden_state, total_loss, recurrent_loss, loss_per_stage

    def step(self, is_train):
        stages_per_batch = self.get_stages_per_batch()
        if is_train:
            self.scheduler.step()
            epoch = self.scheduler.last_epoch
            lr = self.scheduler.get_lr()[0]
            self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
            loss_per_stage = None
        else:
            epoch = self.scheduler.last_epoch
            loss_per_stage = [[] for _ in range(stages_per_batch)]
        total_loss = 0.0
        timer_model = utility.Timer()
        print_batch = 0
        train_batches = 200000 if self.args.dataset.startswith('imagenet') else (
            100000 if self.args.dataset == 'tiny' else 40000)
        train_batches //= self.args.batch_size
        val_batches = 20000 if self.args.dataset.startswith('imagenet') else (
            10000 if self.args.dataset == 'tiny' else 10000)
        val_batches //= self.args.batch_size
        if self.is_fast_test:
            train_batches = 4
            val_batches = 5
        loader_val = iter(self.loader_val)
        code_size = np.array([])
        conversion = np.log(2) * 3
        self.model.train(is_train)
        with torch.set_grad_enabled(is_train):
            for batch in trange(train_batches if is_train else val_batches,
                                desc='Train Step' if is_train else 'Validation Step', dynamic_ncols=True):
                orig_lr, hr, bpg_size = next(self.loader_train if is_train else loader_val)
                code_size = np.append(code_size, bpg_size.mean().item())
                timer_model.tic()
                orig_lr, hr = self.prepare([orig_lr, hr])
                B, _, H, W = hr.size()
                hidden_state = torch.zeros(B, self.args.n_feats, H, W).to(hr)
                recurrent_loss = 0
                print_batch += 1

                for stage in range(stages_per_batch):
                    packed_res = self.step_stage(is_train, stage, B, H, W, orig_lr, hr, hidden_state,
                                                 total_loss, stages_per_batch, recurrent_loss, loss_per_stage)
                    hidden_state, total_loss, recurrent_loss, loss_per_stage = packed_res

                timer_model.hold()
                if is_train and (batch + 1) % self.args.print_every == 0:
                    total_loss = (total_loss / print_batch / stages_per_batch / conversion) + code_size.mean()
                    self.ckp.write_log('Train:\t{:.12f} in {:.2f}s'.format(total_loss, timer_model.release()))
                    self.ckp.add_scalar('train_loss', total_loss, batch + (epoch - 1) * train_batches)
                    total_loss = 0
                    print_batch = 0
                    code_size = np.array([])
        if not is_train:
            code_size = code_size.mean()
            loss_per_stage = np.array(loss_per_stage).mean(axis=1) / conversion
            weighted_loss_per_stage = loss_per_stage * self.stage_weights
            val_loss = weighted_loss_per_stage.sum() / self.stage_weights.sum() + code_size
            self.ckp.write_log(f'DetailedLoss: {loss_per_stage}')
            self.ckp.write_log(f'val_Loss: {val_loss}, non_normalized: {loss_per_stage.mean() + code_size}')
            self.ckp.add_scalar('val_loss', val_loss, epoch * train_batches)
            for i in range(len(loss_per_stage)):
                self.ckp.add_scalar(f'detailed_loss_{i}', loss_per_stage[i], epoch * train_batches)
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            return is_best, val_loss

    def prepare(self, l):
        def _prepare(tensor):
            if not self.args.cpu:
                tensor = tensor.cuda()
            return tensor

        return [_prepare(_l) for _l in l]

    def terminate(self):
        self.current_epoch = self.scheduler.last_epoch + 1
        return self.scheduler.last_epoch >= self.args.epochs
