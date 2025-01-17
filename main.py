import os
from os import path, makedirs
import argparse
import time
import math
import builtins
import subprocess

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms

from simsiam.loader import TwoCropsTransform
from simsiam.model_factory import SimSiam
from simsiam.criterion import SimSiamLoss
from simsiam.validation import KNNValidation

parser = argparse.ArgumentParser('arguments for training')
parser.add_argument('--data_root', type=str, help='path to dataset directory')
parser.add_argument('--exp_dir', type=str, help='path to experiment directory')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
parser.add_argument('--trial', type=str, default='1', help='trial id')
parser.add_argument('--img_dim', default=32, type=int)

parser.add_argument('--is_cluster', default=False, action='store_true')
parser.add_argument('--distributed', default=False, action='store_true')

parser.add_argument('--arch', default='resnet18', help='model name is used for training')

parser.add_argument('--feat_dim', default=2048, type=int, help='feature dimension')
parser.add_argument('--num_proj_layers', type=int, default=2, help='number of projection layer')
parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
parser.add_argument('--epochs', type=int, default=800, help='number of training epochs')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--loss_version', default='simplified', type=str,
                    choices=['simplified', 'original'],
                    help='do the same thing but simplified version is much faster. ()')
parser.add_argument('--print_freq', default=10, type=int, help='print frequency')
parser.add_argument('--eval_freq', default=10, type=int, help='evaluate model frequency')
parser.add_argument('--save_freq', default=50, type=int, help='save model frequency')
parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint')

parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

args = parser.parse_args()


def main():
    if args.is_cluster and args.distributed:
        # local rank on the current node / global rank
        local_rank = int(os.environ['SLURM_LOCALID'])
        global_rank = int(os.environ['SLURM_PROCID'])
        # number of processes / GPUs per node
        world_size = int(os.environ['SLURM_NTASKS'])
        # define master address and master port
        hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
        master_addr = hostnames.split()[0].decode('utf-8')
        # set environment variables for 'env://'
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(29500)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(global_rank)
        os.environ['LOCAL_RANK'] = str(local_rank)

    if args.distributed:
        torch.distributed.init_process_group(backend='nccl')
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        is_master = rank == 0
        if is_master:

            def print_pass(*args, **kwargs):
                pass
            builtins.print = print_pass
    else:
        world_size = 1
        rank = 0
        local_rank = 0
        is_master = True
    torch.cuda.set_device(local_rank)

    if is_master:
        if not path.exists(args.exp_dir):
            makedirs(args.exp_dir)

    trial_dir = path.join(args.exp_dir, args.trial)
    if is_master:
        logger = SummaryWriter(trial_dir)
    print(vars(args))

    if args.dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
    else:
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(args.img_dim, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_set = (datasets.CIFAR10 if args.dataset == 'cifar10' else datasets.CIFAR100)(
        root=args.data_root,
        train=True,
        download=not args.is_cluster,
        transform=TwoCropsTransform(train_transforms),
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set) if args.distributed else None
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=int(args.batch_size / world_size),
        shuffle=not args.distributed,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        sampler=train_sampler,
    )

    model = SimSiam(args)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) if args.distributed else model
    model.to('cuda', non_blocking=True)

    args.learning_rate = args.learning_rate * args.batch_size / 256
    optimizer = optim.SGD(model.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank) if args.distributed else model
    criterion = SimSiamLoss()

    start_epoch = 1
    if args.resume is not None:
        if path.isfile(args.resume):
            start_epoch, model, optimizer = load_checkpoint(model, optimizer, args.resume)
            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, start_epoch))
        else:
            print("No checkpoint found at '{}'".format(args.resume))

    # routine
    best_acc = 0.0
    validation = KNNValidation(args, model.module.online_encoder if args.distributed else model.online_encoder)
    for epoch in range(start_epoch, args.epochs+1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)
        print("Training...")

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch, args)
        if is_master:
            logger.add_scalar('Loss/train', train_loss, epoch)

        if (epoch % args.eval_freq == 0) and is_master:
            print("Validating...")
            val_top1_acc = validation.eval()
            print('Top1: {}'.format(val_top1_acc))

            # save the best model
            if val_top1_acc > best_acc:
                best_acc = val_top1_acc

                save_checkpoint(epoch, model, optimizer, best_acc,
                                path.join(trial_dir, '{}_best.pth'.format(args.trial)),
                                'Saving the best model!')
            logger.add_scalar('Acc/val_top1', val_top1_acc, epoch)

        # save the model
        if (epoch % args.save_freq == 0) and is_master:
            save_checkpoint(epoch, model, optimizer, val_top1_acc,
                            path.join(trial_dir, 'ckpt_epoch_{}_{}.pth'.format(epoch, args.trial)),
                            'Saving...')

    print('Best accuracy:', best_acc)

    # save model
    if is_master:
        save_checkpoint(epoch, model, optimizer, val_top1_acc,
                        path.join(trial_dir, '{}_last.pth'.format(args.trial)),
                        'Saving the model at the last epoch.')


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    step = (epoch - 1) * len(train_loader)
    for i, (images, _) in enumerate(train_loader):

        images[0] = images[0].cuda('cuda', non_blocking=True)
        images[1] = images[1].cuda('cuda', non_blocking=True)

        # compute output
        mm = adjust_mm(0.996, step, args.epochs * len(train_loader))
        step += 1
        outs = model(im_aug1=images[0], im_aug2=images[1], mm=mm)
        loss = criterion(outs['z1'], outs['z2'], outs['p1'], outs['p2'])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        losses.update(loss.item(), images[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.learning_rate
    # cosine lr schedule
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_mm(init_mm: float, step: int, max_step: int) -> float:
    return 1 - (1 - init_mm) * (math.cos(math.pi*step/max_step)+1)/2


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_checkpoint(epoch, model, optimizer, acc, filename, msg):
    state = {
        'epoch': epoch,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'top1_acc': acc
    }
    torch.save(state, filename)
    print(msg)


def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename, map_location='cuda:0')
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return start_epoch, model, optimizer


if __name__ == '__main__':
    main()
