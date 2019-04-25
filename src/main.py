import argparse
import shutil
import os
import time

import numpy as np
import quaternion
import matplotlib.pyplot as plt

import torch
import torchvision

from torch.backends import cudnn
from torch.utils.data import DataLoader

from posenet import PoseNet, PoseNetCriterion, PoseDataset


#DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/cambridge')
DATA_DIR = 'd:\\projects\\densepose\\data\\cambridge'


# Basic structure taken from https://github.com/pytorch/examples/imagenet/main.py

model_names = sorted(name for name in torchvision.models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision.models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch PoseNet Training')
parser.add_argument('--data', '-d', metavar='DIR', default=DATA_DIR,
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='densenet161',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: densenet161)')
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
parser.add_argument('--weight-decay', '--wd', default=0.5, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default=True, action='store_true',
                    help='use pre-trained model')


def main():
    global args
    args = parser.parse_args()

    # if dont call torch.cuda.current_device(), fails later with
    #   "RuntimeError: cuda runtime error (30) : unknown error at ..\aten\src\THC\THCGeneral.cpp:87"
    torch.cuda.current_device()
    use_cuda = torch.cuda.is_available() and True
    device = torch.device("cuda:0" if use_cuda else "cpu")
    cudnn.benchmark = False  # uses extra mem if True

    model = PoseNet(arch=args.arch, num_features=args.features, dropout=0, pretrained=True,
                    track_running_stats=False, cache_dir='d:\\projects\\densepose\\data\\models')

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
    train_loader = DataLoader(
        PoseDataset(args.data, 'dataset_train.txt', random_crop=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = DataLoader(
        PoseDataset(args.data, 'dataset_test.txt', random_crop=False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and pptimizer
    criterion = PoseNetCriterion(stereo=False, learn_beta=True, sx=0.0, sq=-3.0)

    # evaluate model only
    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    # initialize optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                     weight_decay=args.weight_decay)
    else:
        assert False, 'Invalid optimizer: %s' % args.optimizer

    model.to(device)
    criterion.to(device)

    # training loop
    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device)

        # evaluate on validation set
        loss = validate(val_loader, model, criterion, device)
        print('=====\n')

        # remember best loss and save checkpoint
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_loss': loss,
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, device, validate_only=False):
    batch_time = Meter()
    losses = Meter()
    positions = Meter()
    orientations = Meter()

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

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and optimize params
        if not validate_only:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure accuracy and record loss
        pos, orient = accuracy(output.data, target)
        positions.update(pos)
        orientations.update(orient)
        losses.update(loss.data)
        mem_usage = torch.cuda.max_memory_allocated() / 1024 ** 3

        # tried to clean up in hope that frees gpu mem before next iteration => did not help
        #optimizer.zero_grad()
        #del loss
        #del output

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print((('Test [{1}/{2}]' if validate_only else 'Epoch: [{0}][{1}/{2}]\t') +
                  ' GPU Mem: {mem_usage:.2f}GB\t'
                  ' Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  ' Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                  ' Position: {pos.val:.3f} ({pos.median:.3f})\t'
                  ' Orientation: {orient.val:.3f} ({orient.median:.3f})').format(
                   epoch, i, len(train_loader), mem_usage=mem_usage, batch_time=batch_time,
                   loss=losses, pos=positions, orient=orientations))

    return losses.avg


def validate(val_loader, model, criterion, device):
    with torch.no_grad():
        result = train(val_loader, model, criterion, None, None, device, validate_only=True)
    return result


def accuracy(output, target):
    """ Computes position and orientation accuracy """
    # batch_size = target.size(0)
    output_np = output.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    err_pos = np.linalg.norm(output_np[:, :3] - target_np[:, :3], axis=1)
    err_orient = angle_between_q(output_np[:, 3:], target_np[:, 3:])
    return err_pos, err_orient


def angle_between_q(q1r, q2r):
    # from  https://chrischoy.github.io/research/measuring-rotation/
    q1 = quaternion.from_float_array(q1r)
    q2 = quaternion.from_float_array(q2r)
    qd = q1.conj()*q2
    err_rad = 2*np.arccos([q.normalized().w for q in qd])
    return np.abs((np.degrees(err_rad) + 180) % 360 - 180)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Meter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    @property
    def val(self):
        return self.values[-1]

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
        self.values = []

    def update(self, val):
        if torch.is_tensor(val):
            val = val.detach().cpu().numpy()

        try:
            self.values.extend(val)
        except:
            self.values.append(val)



if __name__ == '__main__':
    main()
