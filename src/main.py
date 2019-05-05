import argparse
import random
import shutil
import os
import time
import csv

import numpy as np
import quaternion
import matplotlib.pyplot as plt

import torch
import torchvision

from torch.backends import cudnn
from torch.utils.data import DataLoader

from posenet import PoseNet, PoseDataset


#DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/cambridge')
DATA_DIR = 'd:\\projects\\densepose\\data\\cambridge\\StMarysChurch'
CACHE_DIR = 'd:\\projects\\densepose\\data\\models'
RND_SEED = 10

# Basic structure taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py

model_names = sorted(name for name in torchvision.models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision.models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch PoseNet Training')
parser.add_argument('--data', '-d', metavar='DIR', default=DATA_DIR,
                    help='path to dataset')
parser.add_argument('--cache', metavar='DIR', default=CACHE_DIR,
                    help='path to cache dir')
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
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                     help='momentum (for SGD optimizer only)')
parser.add_argument('--optimizer', '-o', default='adam', type=str, metavar='OPT',
                     help='optimizer [adam|sgd]', choices=('sgd', 'adam'))
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
parser.add_argument('--warmup', default=0, type=int, metavar='N',
                    help='number of warmup epochs where only newly added layers are trained')
parser.add_argument('--split-opt-params', default=False, action='store_true',
                    help='use different optimization params for bias, weight and loss function params')
parser.add_argument('--excl-bn', default=False, action='store_true',
                    help='exclude batch norm params from optimization')
parser.add_argument('--name', '--pid', default='', type=str, metavar='NAME',
                    help='experiment name for out file names')


def main():
    global args
    args = parser.parse_args()

    # if dont call torch.cuda.current_device(), fails later with
    #   "RuntimeError: cuda runtime error (30) : unknown error at ..\aten\src\THC\THCGeneral.cpp:87"
    torch.cuda.current_device()
    use_cuda = torch.cuda.is_available() and True
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # try to get consistent results across runs => fails, but makes runs a bit more similar
    _set_random_seed()

    model = PoseNet(arch=args.arch, num_features=args.features, dropout=args.dropout,
                    pretrained=True, cache_dir=args.cache, loss=args.loss, excl_bn_affine=args.excl_bn,
                    beta=args.beta, sx=args.sx, sq=args.sq)

    # optionally resume from a checkpoint
    best_loss = float('inf')
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    trdata = PoseDataset(args.data, 'dataset_train.txt', random_crop=True)
    train_loader = DataLoader(trdata,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        worker_init_fn=_worker_init_fn)

    val_loader = DataLoader(
        PoseDataset(args.data, 'dataset_test.txt', random_crop=False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        worker_init_fn=_worker_init_fn)

    model.set_target_transform(trdata.target_mean, trdata.target_std)
    model.to(device)

    # evaluate model only
    if args.evaluate:
        validate(val_loader, model)
        return

    # initialize optimizer with warmup param values
    nlr, nwd, olr, owd = (30, 30, 10, 10)
    if args.optimizer == 'sgd':
        raise NotImplementedError('SGD not implemented')
        params = model.params_to_optimize(excl_batch_norm=args.excl_bn)
        optimizer = torch.optim.SGD(params, lr=args.lr * nlr, momentum=args.momentum, weight_decay=args.weight_decay * nwd)
    elif args.optimizer == 'adam':
        if args.split_opt_params:
            new_biases, new_weights, biases, weights, others = model.params_to_optimize(split=True, excl_batch_norm=args.excl_bn)
            optimizer = torch.optim.Adam([
                {'params': new_biases, 'lr': args.lr * nlr * 2, 'weight_decay': args.weight_decay * nwd * 0.0, 'eps': 0.1},
                {'params': new_weights, 'lr': args.lr * nlr * 1, 'weight_decay': args.weight_decay * nwd * 1, 'eps': 0.1},
                {'params': biases, 'lr': args.lr * olr * 2, 'weight_decay': args.weight_decay * owd * 0.0, 'eps': 0.1},
                {'params': weights, 'lr': args.lr * olr * 1, 'weight_decay': args.weight_decay * owd * 1, 'eps': 0.1},
                {'params': others, 'lr': 0, 'weight_decay': 0, 'eps': 0.1},
            ])
        else:
            params = model.params_to_optimize(excl_batch_norm=args.excl_bn)
            optimizer = torch.optim.Adam(params, lr=args.lr * nlr, weight_decay=args.weight_decay * nwd, eps=0.1)
    else:
        assert False, 'Invalid optimizer: %s' % args.optimizer

    # training loop
    stats = np.zeros((args.epochs, 11))
    for epoch in range(args.start_epoch, args.epochs):
        if epoch == args.warmup:
            end_warmup(optimizer)

        # train for one epoch
        lss, pos, ori = train(train_loader, model, optimizer, epoch, device)
        stats[epoch, :6] = [epoch, lss.avg, pos.avg, pos.median, ori.avg, ori.median]

        # evaluate on validation set
        if (epoch+1) % args.test_freq == 0:
            lss, pos, ori = validate(val_loader, model, device)
            stats[epoch, 6:] = [lss.avg, pos.avg, pos.median, ori.avg, ori.median]

            # remember best loss and save checkpoint
            is_best = ori.median < best_loss
            best_loss = min(ori.median, best_loss)

            if is_best:
                _save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_loss': lss.avg,
                }, True)
        else:
            is_best = False

        if (epoch+1) % args.save_freq == 0 and not is_best:
            _save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_loss': lss.avg,
            }, False)

        print('=====\n')
    _save_log(stats)


def train(train_loader, model, optimizer, epoch, device, validate_only=False):
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
    for i, (input, target) in enumerate(train_loader):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # measure elapsed data loading time
        data_time.update(time.time() - end)
        end = time.time()

        # compute output
        output = model(input)
        loss = model.cost(output, target)

        # compute gradient and optimize params
        if not validate_only:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure accuracy and record loss
        with torch.no_grad():
            output = output[0] if isinstance(output, (list, tuple)) else output
            pos, orient = accuracy(output, target)

        positions.update(pos)
        orientations.update(orient)
        losses.update(loss.data)

        # measure elapsed processing time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0 or i+1 == len(train_loader):
            print((('Test [{1}/{2}]' if validate_only else 'Epoch: [{0}][{1}/{2}]\t') +
                  ' Load: {data_time.pop_recent:.3f} ({data_time.avg:.3f})\t'
                  ' Proc: {batch_time.pop_recent:.3f} ({batch_time.avg:.3f})\t'
                  ' Loss: {loss.pop_recent:.4f} ({loss.avg:.4f})\t'
                  ' Pos: {pos.pop_recent:.3f} ({pos.median:.3f})\t'
                  ' Ori: {orient.pop_recent:.3f} ({orient.median:.3f})'
                  ' CF: ({cost_sx:.3f}, {cost_sq:.3f})').format(
                   epoch, i+1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, pos=positions, orient=orientations,
                   cost_sx=float(model.cost_fn.sx.data), cost_sq=float(model.cost_fn.sq.data)))

    return losses, positions, orientations


def validate(val_loader, model, device):
    with torch.no_grad():
        result = train(val_loader, model, None, None, device, validate_only=True)
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
    return (filename[:-ext] + '_' + args.name + filename[-ext:]) if len(args.name) > 0 else filename


def _save_log(stats, filename='stats.csv'):
    with open(_filename_pid(filename), 'w', newline='') as fh:
        w = csv.writer(fh, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        w.writerow(['epoch', 'tr_loss', 'tr_err_v_avg', 'tr_err_v_med', 'tr_err_q_avg', 'tr_err_q_med', 'tst_loss', 'tst_err_v_avg', 'tst_err_v_med', 'tst_err_q_avg', 'tst_err_q_med'])
        for row in stats:
            w.writerow(row)


def _save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, _filename_pid(filename))
    if is_best:
        shutil.copyfile(_filename_pid(filename), _filename_pid('model_best.pth.tar'))


def end_warmup(optimizer):
    if len(optimizer.param_groups) == 1:
        optimizer.param_groups[0]['lr'] = args.lr
        optimizer.param_groups[0]['weight_decay'] = args.weight_decay
        return

    # new biases
    optimizer.param_groups[0]['lr'] = args.lr * 2
    optimizer.param_groups[0]['weight_decay'] = args.weight_decay * 0.0

    # new weights
    optimizer.param_groups[1]['lr'] = args.lr
    optimizer.param_groups[1]['weight_decay'] = args.weight_decay

    # old biases
    optimizer.param_groups[2]['lr'] = args.lr * 2
    optimizer.param_groups[2]['weight_decay'] = args.weight_decay * 0.0

    # old weights
    optimizer.param_groups[3]['lr'] = args.lr
    optimizer.param_groups[3]['weight_decay'] = args.weight_decay

    # others (sx, sq)
    optimizer.param_groups[4]['lr'] = args.lr
    optimizer.param_groups[4]['weight_decay'] = 0


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
