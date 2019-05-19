import argparse
import random
import shutil
import os
import time
import csv

import numpy as np
import sys

import torch
import torchvision

from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from posenet import PoseNet, PoseDataset

# random seed used
RND_SEED = 10

# for my own convenience
DEFAULT_DATA_DIR = 'd:\\projects\\densepose\\data\\cambridge\\StMarysChurch'
DEFAULT_CACHE_DIR = 'd:\\projects\\densepose\\data\\models'
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')

# Basic structure inspired by https://github.com/pytorch/examples/blob/master/imagenet/main.py

model_names = sorted(name for name in torchvision.models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision.models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch PoseNet Training')
parser.add_argument('--data', '-d', metavar='DIR', default=DEFAULT_DATA_DIR,
                    help='path to dataset')
parser.add_argument('--cache', metavar='DIR', default=DEFAULT_CACHE_DIR,
                    help='path to cache dir')
parser.add_argument('--output', metavar='DIR', default=DEFAULT_OUTPUT_DIR,
                    help='path to output dir')
parser.add_argument('--arch', '-a', metavar='ARCH', default='googlenet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: googlenet)')
parser.add_argument('-n', '--features', default=2048, type=int, metavar='N',
                    help='number of localization features (default: 2048)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1500, type=int, metavar='N',
                    help='number of total epochs to run (default: 1500)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 8)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate (default: 1e-4)')
parser.add_argument('--optimizer', '-o', default='adam', type=str, metavar='OPT',
                     help='optimizer, only [adam] currently available', choices=('adam',))
parser.add_argument('--weight-decay', '--wd', default=0.0005, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--dropout', '--do', default=0, type=float,
                    metavar='R', help='dropout ratio (default: 0)')
parser.add_argument('--loss', default='L1', type=str,
                    metavar='L', help='loss metric [L1|MSE] (default: L1)')
parser.add_argument('--beta', default=250, type=float,
                    metavar='B', help='fixed orientation loss weight, set to zero '
                    'to learn sx and sq instead (default: 250)')
parser.add_argument('--sx', default=0.0, type=float,
                    metavar='SX', help='initial position loss weight (default: 0.0)')
parser.add_argument('--sq', default=-6.0, type=float,
                    metavar='SQ', help='initial orientation loss weight (default: -6.0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--test-freq', '--tf', default=1, type=int,
                    metavar='N', help='test frequency (default: 1)')
parser.add_argument('--save-freq', '--sf', default=10, type=int,
                    metavar='N', help='save frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default=True, action='store_true',
                    help='use pre-trained model')
parser.add_argument('--split-opt-params', default=False, action='store_true',
                    help='use different optimization params for bias, weight and loss function params')
parser.add_argument('--excl-bn', default=False, action='store_true',
                    help='exclude batch norm params from optimization')
parser.add_argument('--adv-tr-eps', default=0, type=float, metavar='eps',
                    help='use adversarial training with given epsilon')
parser.add_argument('--center-crop', default=False, action='store_true',
                    help='use center crop instead of random crop for training')
parser.add_argument('--early-stopping', default=0, type=int, metavar='N',
                    help='stop training, if loss on validation set does not decrease for this many epochs')
parser.add_argument('--name', '--pid', default='', type=str, metavar='NAME',
                    help='experiment name for out file names')


def main():
    global args
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # if don't call torch.cuda.current_device(), fails later with
    #   "RuntimeError: cuda runtime error (30) : unknown error at ..\aten\src\THC\THCGeneral.cpp:87"
    torch.cuda.current_device()
    use_cuda = torch.cuda.is_available() and True
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # try to get consistent results across runs
    #   => currently still fails, however, makes runs a bit more consistent
    _set_random_seed()

    # create model
    model = PoseNet(arch=args.arch, num_features=args.features, dropout=args.dropout,
                    pretrained=True, cache_dir=args.cache, loss=args.loss, excl_bn_affine=args.excl_bn,
                    beta=args.beta, sx=args.sx, sq=args.sq)

    # create optimizer
    #  - currently only Adam supported
    if args.optimizer == 'adam':
        eps = 0.1
        if args.split_opt_params:
            new_biases, new_weights, biases, weights, others = model.params_to_optimize(split=True, excl_batch_norm=args.excl_bn)
            optimizer = torch.optim.Adam([
                {'params': new_biases, 'lr': args.lr * 2, 'weight_decay': 0.0, 'eps': eps},
                {'params': new_weights, 'lr': args.lr, 'weight_decay': args.weight_decay, 'eps': eps},
                {'params': biases, 'lr': args.lr * 2, 'weight_decay': 0.0, 'eps': eps},
                {'params': weights, 'lr': args.lr, 'weight_decay': args.weight_decay, 'eps': eps},
                {'params': others, 'lr': 0, 'weight_decay': 0, 'eps': eps},
            ])
        else:
            params = model.params_to_optimize(excl_batch_norm=args.excl_bn)
            optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay, eps=eps)
    else:
        assert False, 'Invalid optimizer: %s' % args.optimizer

    # optionally resume from a checkpoint
    best_loss = float('inf')
    best_epoch = -1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_epoch = checkpoint['best_epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # define overall training dataset, set output normalization, load model to gpu
    all_tr_data = PoseDataset(args.data, 'dataset_train.txt', random_crop=not args.center_crop)
    model.set_target_transform(all_tr_data.target_mean, all_tr_data.target_std)
    model.to(device)

    # split overall training data to training and validation sets
    # validation set is used for early stopping, or possibly in future for hyper parameter optimization
    lengths = [round(len(all_tr_data) * 0.75), round(len(all_tr_data) * 0.25)]
    tr_data, val_data = torch.utils.data.random_split(all_tr_data, lengths)

    # define data loaders
    train_loader = DataLoader(tr_data, batch_size=args.batch_size, num_workers=args.workers,
                              shuffle=True, pin_memory=True, worker_init_fn=_worker_init_fn)

    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.workers,
                            shuffle=False, pin_memory=True, worker_init_fn=_worker_init_fn)

    test_loader = DataLoader(PoseDataset(args.data, 'dataset_test.txt', random_crop=False),
                             batch_size=args.batch_size, num_workers=args.workers,
                             shuffle=False, pin_memory=True, worker_init_fn=_worker_init_fn)

    # evaluate model only
    if args.evaluate:
        validate(test_loader, model)
        return

    # training loop
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        lss, pos, ori = process(train_loader, model, optimizer, epoch, device, adv_tr_eps=args.adv_tr_eps)
        stats = np.zeros(16)
        stats[:6] = [epoch, lss.avg, pos.avg, pos.median, ori.avg, ori.median]

        # evaluate on validation set
        if (epoch+1) % args.test_freq == 0:
            lss, pos, ori = validate(val_loader, model, device)
            stats[6:11] = [lss.avg, pos.avg, pos.median, ori.avg, ori.median]

            # remember best loss and save checkpoint
            is_best = lss.avg < best_loss
            best_epoch = epoch if is_best else best_epoch
            best_loss = lss.avg if is_best else best_loss

            # save best model
            if is_best:
                _save_checkpoint({
                    'epoch': epoch + 1,
                    'best_epoch': best_epoch,
                    'best_loss': best_loss,
                    'arch': args.arch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, True)
        else:
            is_best = False

        # maybe save a checkpoint even if not best model
        if (epoch+1) % args.save_freq == 0 and not is_best:
            _save_checkpoint({
                'epoch': epoch + 1,
                'best_epoch': best_epoch,
                'best_loss': best_loss,
                'arch': args.arch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, False)

        # evaluate on test set if best yet result on validation set
        if is_best:
            lss, pos, ori = validate(test_loader, model, device)
            stats[11:] = [lss.avg, pos.avg, pos.median, ori.avg, ori.median]

        # add row to log file
        _save_log(stats, epoch == 0)

        # early stopping
        if args.early_stopping > 0 and epoch - best_epoch >= args.early_stopping:
            print('=====\nEARLY STOPPING CRITERION MET (%d epochs since best validation loss)' % args.early_stopping)
            break

        print('=====\n')

    if epoch+1 == args.epochs:
        print('MAX EPOCHS (%d) REACHED' % args.epochs)
    print('BEST VALIDATION LOSS: %.3f' % best_loss)


def process(loader, model, optimizer, epoch, device, validate_only=False, adv_tr_eps=0):
    data_time = Meter()
    batch_time = Meter()
    losses = Meter()
    positions = Meter(median=True)
    orientations = Meter(median=True)

    if validate_only:
        # switch to evaluate mode
        model.eval()
    else:
        # switch to train mode
        model.train()

    end = time.time()
    for i, (input, target) in enumerate(loader):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # measure elapsed data loading time
        data_time.update(time.time() - end)
        end = time.time()

        if adv_tr_eps > 0:
            input.requires_grad = True

        # compute output
        output = model(input)
        loss = model.cost(output, target)

        # measure accuracy and record loss
        with torch.no_grad():
            output = output[0] if isinstance(output, (list, tuple)) else output
            pos, orient = accuracy(output, target)
            positions.update(pos)
            orientations.update(orient)
            losses.update(loss.data)

        # compute gradient and optimize params
        if not validate_only:
            optimizer.zero_grad()
            loss.backward()

            if adv_tr_eps > 0:
                # adversarial training sample
                alt_input = input + adv_tr_eps * input.grad.data.sign()
                alt_output = model(alt_input)
                alt_loss = model.cost(alt_output, target)
                alt_loss.backward()

            optimizer.step()

        # measure elapsed processing time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0 or i+1 == len(loader):
            print((('Test [{1}/{2}]' if validate_only else 'Epoch: [{0}][{1}/{2}]\t') +
                  ' Load: {data_time.pop_recent:.3f} ({data_time.avg:.3f})\t'
                  ' Proc: {batch_time.pop_recent:.3f} ({batch_time.avg:.3f})\t'
                  ' Loss: {loss.pop_recent:.4f} ({loss.avg:.4f})\t'
                  ' Pos: {pos.pop_recent:.3f} ({pos.median:.3f})\t'
                  ' Ori: {orient.pop_recent:.3f} ({orient.median:.3f})'
                  ' CF: ({cost_sx:.3f}, {cost_sq:.3f})').format(
                   epoch, i+1, len(loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, pos=positions, orient=orientations,
                   cost_sx=float(model.cost_fn.sx.data), cost_sq=float(model.cost_fn.sq.data)))

    return losses, positions, orientations


def validate(test_loader, model, device):
    with torch.no_grad():
        result = process(test_loader, model, None, None, device, validate_only=True)
    return result


def accuracy(output, target):
    """ Computes position and orientation accuracy """
    err_pos = torch.sum((output[:, :3] - target[:, :3])**2, dim=1)**(1/2)
    err_orient = _angle_between_q(output[:, 3:], target[:, 3:])
    return err_pos, err_orient


def _angle_between_q(q1, q2):
    # from https://github.com/hazirbas/poselstm-pytorch/blob/master/models/posenet_model.py
    abs_distance = torch.clamp(torch.abs(torch.sum(q2.mul(q1), dim=1)), 0, 1)
    ori_err = 2 * 180 / np.pi * torch.acos(abs_distance)
    return ori_err


def _set_random_seed(seed=RND_SEED): #, fanatic=False):
    # doesnt work even if fanatic & use_cuda
    # if fanatic:
    #     # if not disabled, still some variation between runs, however, makes training painfully slow
    #     cudnn.enabled = False       # ~double time
    # if use_cuda:
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True      # 7% slower
    cudnn.benchmark = False         # also uses extra mem if True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _worker_init_fn(id):
    np.random.seed(RND_SEED)


def _filename_pid(filename):
    ext = len(filename) - max(filename.find('.'), 0)
    filename = (filename[:-ext] + '_' + args.name + filename[-ext:]) if len(args.name) > 0 else filename
    return os.path.join(args.output, filename)


def _save_log(stats, write_header, filename='stats.csv'):
    with open(_filename_pid(filename), 'a', newline='') as fh:
        w = csv.writer(fh, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # maybe write header
        if write_header:
            w.writerow([' '.join(sys.argv)])
            w.writerow(['epoch', 'tr_loss', 'tr_err_v_avg', 'tr_err_v_med', 'tr_err_q_avg', 'tr_err_q_med',
                                 'val_loss', 'val_err_v_avg', 'val_err_v_med', 'val_err_q_avg', 'val_err_q_med',
                                 'tst_loss', 'tst_err_v_avg', 'tst_err_v_med', 'tst_err_q_avg', 'tst_err_q_med'])

        # write stats one epoch at a time
        w.writerow(stats)


def _save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, _filename_pid(filename))
    if is_best:
        shutil.copyfile(_filename_pid(filename), _filename_pid('model_best.pth.tar'))


class Meter(object):
    """ Stores current values and calculates stats """
    def __init__(self, median=False):
        self.default_median = median
        self.reset()

    @property
    def pop_recent(self):
        if self.default_median:
            val = np.median(self.recent_values)
        else:
            val = np.mean(self.recent_values)
        self.recent_values.clear()
        return val

    @property
    def sum(self):
        return np.sum(self.values)

    @property
    def count(self):
        return len(self.values)

    @property
    def avg(self):
        return np.mean(self.values)

    @property
    def median(self):
        return np.median(self.values)

    def reset(self):
        self.recent_values = []
        self.values = []

    def update(self, val):
        if torch.is_tensor(val):
            val = val.detach().cpu().numpy()
        if isinstance(val, (list, tuple)):
            val = np.array(val)
        if not isinstance(val, np.ndarray):
            val = np.array([val])

        self.recent_values.extend(val)
        self.values.extend(val)


if __name__ == '__main__':
    main()
