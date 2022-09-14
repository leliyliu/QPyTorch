from __future__ import print_function

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import numpy as np
import os
import shutil
import argparse
import time
import logging
import wandb 
import socket
import plotly.express as px
import plotly.figure_factory as ff 
import copy 

import models
from modules.data import *


import util_swa

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__')
                     and callable(models.__dict__[name])
                     )


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch CIFAR training in a fixed-point low-precision way!')
    parser.add_argument("--team_name", type=str,  default='leliy-ict')
    parser.add_argument("--project_name", type=str, default='low-precision-fixed-point-training')
    parser.add_argument("--experiment_name", type=str, default='distribution-analysis')
    parser.add_argument("--scenario_name", type=str, default='difference-in-weight')
    parser.add_argument('--dir', help='annotate the working directory')
    parser.add_argument('--cmd', choices=['train', 'test'], default='train')
    parser.add_argument('--arch', metavar='ARCH', default='cifar10_resnet_38',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: cifar10_resnet_38)')
    parser.add_argument('--dataset', '-d', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='dataset choice')
    parser.add_argument('--datadir', default='/home/leliy/datasets/cifar-100', type=str,
                        help='path to dataset')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4 )')
    parser.add_argument('--epochs', default=200, type=int,
                        help='number of total iterations (default: 64,000)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual iter number (useful on restarts)')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='mini-batch size (default: 128)')
    parser.add_argument('--lr_schedule', default='piecewise', type=str,
                        help='learning rate schedule')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--print_freq', default=50, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--profile_freq', default=200, type=int,
                        help='profile frequency (default: 200)')
    parser.add_argument('--distribution_freq', default=5, type=int,
                        help='distribution frequency for changed weight (default: 5)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to  latest checkpoint (default: None)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pretrained model')
    parser.add_argument('--step_ratio', default=0.1, type=float,
                        help='ratio for learning rate deduction')
    parser.add_argument('--warm_up', action='store_true',
                        help='for n = 18, the model needs to warm up for 400 '
                             'iterations')
    parser.add_argument('--save_folder', default='save_checkpoints',
                        type=str,
                        help='folder to save the checkpoints')
    # parser.add_argument('--eval_every', default=390, type=int,
    #                     help='evaluate model every (default: 1000) iterations')

    parser.add_argument('--num_bits', default=0, type=int,
                        help='num bits for weight and activation')
    parser.add_argument('--num_grad_bits', default=0, type=int,
                        help='num bits for gradient')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='debug or not ')
    # parser.add_argument('--num_bits_schedule', default=None, type=int, nargs='*',
    #                     help='schedule for weight/act precision')
    # parser.add_argument('--num_grad_bits_schedule', default=None, type=int, nargs='*',
    #                     help='schedule for grad precision')

    # parser.add_argument('--is_cyclic_precision', action='store_true',
    #                     help='cyclic precision schedule')
    # parser.add_argument('--cyclic_num_bits_schedule', default=None, type=int, nargs='*',
    #                     help='cyclic schedule for weight/act precision')
    # parser.add_argument('--cyclic_num_grad_bits_schedule', default=None, type=int, nargs='*',
    #                     help='cyclic schedule for grad precision')
    # parser.add_argument('--num_cyclic_period', default=1, type=int,
    #                     help='number of cyclic period for precision, same for weights/activation and gradients')

    parser.add_argument('--swa_start', type=float, default=None, help='SWA start step number')
    parser.add_argument('--swa_freq', type=float, default=1170,
                        help='SWA model collection frequency')
    args = parser.parse_args()
    return args

class SaveActivation:
    def __init__(self, name, step):
        self.name = name 
        self.total_step = step 
        self.step = step 
        self.inputs = []
        self.outputs = [] 

    def __call__(self, module, module_in, module_out):
        self.step -= 1
        if self.step == 0: 
            self.step = self.total_step
            inps, _, _ = module_in 
            out = module_out 
            # import ipdb; ipdb.set_trace()
            self.inputs.append(inps.detach().cpu().numpy())
            self.outputs.append(out.detach().cpu().numpy())

    def clear(self):
        self.inputs = []
        self.outputs = [] 

def main():
    args = parse_args()
    # wandb.init(project="low-precision-fixed-point-training",
    #             name='{}-GNR-percentage-0.9'.format(args.arch))
    mode = 'disabled' if args.debug else 'online'
    wandb.init(config=args,
               project=args.project_name,
               entity=args.team_name,
               notes=socket.gethostname(),
               name=args.experiment_name,
               group=args.scenario_name,
               job_type="training",
               reinit=True, 
               mode=mode)
    save_path = args.save_path = os.path.join(args.save_folder, args.arch)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # config logging file
    args.logger_file = os.path.join(save_path, 'log_{}.txt'.format(args.cmd))
    handlers = [logging.FileHandler(args.logger_file, mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)

    if args.cmd == 'train':
        logging.info('start training {}'.format(args.arch))
        run_training(args)

    elif args.cmd == 'test':
        logging.info('start evaluating {} with checkpoints from {}'.format(
            args.arch, args.resume))
        test_model(args)


def run_training(args):
    # create model
    model = models.__dict__[args.arch](args.pretrained)
    basemodel = copy.deepcopy(model)
    model = torch.nn.DataParallel(model).cuda()
    



    # import ipdb; ipdb.set_trace()

    if args.swa_start is not None:
        print('SWA training')
        swa_model = torch.nn.DataParallel(models.__dict__[args.arch](args.pretrained)).cuda()
        swa_n = 0

    else:
        print('SGD training')

    wandb.watch(model, log="all")


    best_prec1 = 0
    best_iter = 0

    best_swa_prec = 0
    best_swa_iter = 0

    if args.resume:
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_iter = checkpoint['iter']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])

            if args.swa_start is not None:
                swa_state_dict = checkpoint['swa_state_dict']
                if swa_state_dict is not None:
                    swa_model.load_state_dict(swa_state_dict)
                swa_n_ckpt = checkpoint['swa_n']
                if swa_n_ckpt is not None:
                    swa_n = swa_n_ckpt
                best_swa_prec_ckpt = checkpoint['best_swa_prec']
                if best_swa_prec_ckpt is not None:
                    best_swa_prec = best_swa_prec_ckpt

            logging.info('=> loaded checkpoint `{}` (iter: {})'.format(
                args.resume, checkpoint['iter']
            ))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    cudnn.benchmark = False

    train_loader = prepare_train_data(dataset=args.dataset,
                                      datadir=args.datadir,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.workers)
    test_loader = prepare_test_data(dataset=args.dataset,
                                    datadir=args.datadir,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)
    if args.swa_start is not None:
        swa_loader = prepare_train_data(dataset=args.dataset,
                                        datadir=args.datadir,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=args.workers)

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)



    # model.module.group{}_layer{}.conv1.weight .grad 
    groups = [1,2,3]
    layers = [2,6,10]

    # act_savings = {}
    # epoch = args.start_epoch
    for epoch in range(args.epochs):
        start = time.time()
    #     for group in groups:
    #         for layer in layers:
    #             # import ipdb;ipdb.set_trace()
    #             name = 'group{}_layer{}'.format(group, layer)
    #             component = dict([*model.named_modules()])['module.'+name].conv1
    #             act_name = 'save_act' + name.replace('.', '_')
    #             act_savings[act_name] = SaveActivation(name, args.profile_freq)
    #             hook_name = 'hook' + name.replace('.', '_')
    #             # act_names.append(act_name)
    #             # hook_names.append(hook_name)
    #             # locals()[hook_name] = component.register_forward_hook(act_savings[act_name])
    #             component.register_forward_hook(act_savings[act_name])
        train_prec1, train_loss, cr = train(args,train_loader,model, criterion, optimizer)
        # for hook in hook_names:
        #     locals().get(hook).remove()
        # hook_names = []
        for group in groups:
            for layer in layers:
                name = 'group{}_layer{}'.format(group, layer)
                component = dict([*model.named_modules()])['module.'+name].conv1
                weight = component.weight.clone().cpu().data.numpy().reshape(-1)
                gradient = component.weight.grad.clone().cpu().data.numpy().reshape(-1)
                weight = np.log2(np.abs(weight) + 1e-10)
                gradient = np.log2(np.abs(gradient) + 1e-10)
                # act_name = 'save_act' + name.replace('.', '_')
                # input_array = act_savings[act_name].inputs 
                # output_array = act_savings[act_name].outputs
                # print('input length: {}, output length: {}'.format(len(input_array), len(output_array)))
                # input_array = np.array(input_array).reshape(-1)
                # output_array = np.array(output_array).reshape(-1)
                # input_array = np.log2(np.abs(input_array) + 1e-10)
                # output_array = np.log2(np.abs(output_array) + 1e-10)
                # act_savings[act_name].clear()
                # fig = ff.create_distplot([weight, gradient, input_array, output_array], ['Weight', 'Weight Gradients', 'Input Activation', 'Output Actiation'], bin_size=0.5, show_rug=False)
                fig = ff.create_distplot([weight, gradient], ['Weight', 'Weight Gradients'], bin_size=0.5, show_rug=False)
                wandb.log({'{}-profile'.format(name):wandb.Plotly(fig)}, step=epoch)

        if ((epoch+1) % args.distribution_freq) == 0:
            print('record weight difference of trained models !!!')
            for group in groups:
                for layer in layers:
                    name = 'group{}_layer{}'.format(group, layer)
                    base_component = dict([*basemodel.named_modules()])[name].conv1
                    component = dict([*model.named_modules()])['module.'+name].conv1
                    ori_distribution = base_component.weight.clone().cpu().data.numpy().reshape(-1)
                    cur_distribution = component.weight.clone().cpu().data.numpy().reshape(-1)

                    fig = ff.create_distplot([ori_distribution - cur_distribution], ['Weight Difference during training'], show_hist=False)
                    # fig.write_image('results/{}-weight-difference-{}'.format(name, int((epoch+1) / args.distribution_freq)))
                    wandb.log({'{}-weight-difference'.format(name):wandb.Plotly(fig)}, step=epoch)

        validate_prec1, validate_loss = validate(args, test_loader, model, criterion, epoch)
        adjust_learning_rate(args, optimizer, epoch)

        wandb.log({'train_loss': train_loss, 'val_loss': validate_loss, 'train_acc': train_prec1, 'val_acc': validate_prec1, "lr": optimizer.param_groups[0]["lr"],
            'epoch_time': time.time()-start}, step=epoch)

        is_best = validate_prec1 > best_prec1
        if is_best:
            best_prec1 = validate_prec1
            best_epoch = epoch

            print("Current Best Prec@1: ", best_prec1)
            print("Current Best Epoch: ", best_epoch)
            print("Current cr val: {}, cr avg: {}".format(cr.val, cr.avg))


            # checkpoint_path = os.path.join(args.save_path, 'checkpoint_{:05d}_{:.2f}.pth.tar'.format(epoch, validate_prec1))
            # save_checkpoint({
            #     'epoch': epoch,
            #     'arch': args.arch,
            #     'state_dict': model.state_dict(),
            #     'best_prec1': best_prec1,
            #     'swa_state_dict': swa_model.state_dict() if args.swa_start is not None else None,
            #     'swa_n': swa_n if args.swa_start is not None else None,
            #     'best_swa_prec': best_swa_prec if args.swa_start is not None else None,
            # },
            #     is_best, filename=checkpoint_path)
            # shutil.copyfile(checkpoint_path, os.path.join(args.save_path,
            #                                                 'checkpoint_latest'
            #                                                 '.pth.tar'))
    wandb.save("wandb-{}-{}-{}.h5".format(args.arch, args.experiment_name, args.scenario_name))

def train(args, train_loader, model, criterion, optimizer):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    cr = AverageMeter()

    end = time.time()
    for batch_idx, (input, target) in enumerate(train_loader):
        # measuring data loading time
        data_time.update(time.time() - end)

        # model.train()
        # cyclic_period = int(args.iters / args.num_cyclic_period)
        # cyclic_adjust_precision(args, i, cyclic_period)

        fw_cost = args.num_bits * args.num_bits / 32 / 32
        eb_cost = args.num_bits * args.num_grad_bits / 32 / 32
        gc_cost = eb_cost
        cr.update((fw_cost + eb_cost + gc_cost) / 3)
        target = target.squeeze().long().cuda()
        input_var = Variable(input).cuda()
        target_var = Variable(target).cuda()

        # compute output
        output = model(input_var, args.num_bits, args.num_grad_bits)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print log
        if batch_idx % args.print_freq == 0:
            logging.info("Num bit {}\t"
                            "Num grad bit {}\t".format(args.num_bits, args.num_grad_bits))
            logging.info("Iter: [{0}/{1}]\t"
                            "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                            "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                            "Loss {loss.val:.3f} ({loss.avg:.3f})\t"
                            "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                            "Training FLOPS ratio: {cr.val:.6f} ({cr.avg:.6f})\t".format(
                batch_idx,
                len(train_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                top1=top1,
                cr=cr)
            )


            # if args.swa_start is not None and i >= args.swa_start and i % args.swa_freq == 0:
            #     util_swa.moving_average(swa_model, model, 1.0 / (swa_n + 1))
            #     swa_n += 1
            #     util_swa.bn_update(swa_loader, swa_model, args.num_bits, args.num_grad_bits)
            #     prec1 = validate(args, test_loader, swa_model, criterion, i, swa=True)

            #     if prec1 > best_swa_prec:
            #         best_swa_prec = prec1
            #         best_swa_iter = i

            #     print("Current Best SWA Prec@1: ", best_swa_prec)
            #     print("Current Best SWA Iteration: ", best_swa_iter)
    return top1.avg, losses.avg, cr

def validate(args, test_loader, model, criterion, step, swa=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluation mode
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            target = target.squeeze().long().cuda()
            input_var = Variable(input, volatile=True).cuda()
            target_var = Variable(target, volatile=True).cuda()

            # compute output
            output = model(input_var, args.num_bits, args.num_grad_bits)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, = accuracy(output.data, target, topk=(1,))
            top1.update(prec1.item(), input.size(0))
            losses.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if (i % args.print_freq == 0) or (i == len(test_loader) - 1):
                logging.info(
                    'Test: [{}/{}]\t'
                    'Time: {batch_time.val:.4f}({batch_time.avg:.4f})\t'
                    'Loss: {loss.val:.3f}({loss.avg:.3f})\t'
                    'Prec@1: {top1.val:.3f}({top1.avg:.3f})\t'.format(
                        i, len(test_loader), batch_time=batch_time,
                        loss=losses, top1=top1
                    )
                )

        if not swa:
            logging.info('Step {} * Prec@1 {top1.avg:.3f}'.format(step, top1=top1))
        else:
            logging.info('Step {} * SWA Prec@1 {top1.avg:.3f}'.format(step, top1=top1))

    return top1.avg, losses.avg


def test_model(args):
    # create model
    model = models.__dict__[args.arch](args.pretrained)
    model = torch.nn.DataParallel(model).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_iter = checkpoint['iter']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info('=> loaded checkpoint `{}` (iter: {})'.format(
                args.resume, checkpoint['iter']
            ))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    cudnn.benchmark = False
    test_loader = prepare_test_data(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)
    criterion = nn.CrossEntropyLoss().cuda()

    with torch.no_grad():
        prec1 = validate(args, test_loader, model, criterion, args.start_iter)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        save_path = os.path.dirname(filename)
        shutil.copyfile(filename, os.path.join(save_path,
                                               'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def  update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cyclic_adjust_precision(args, _iter, cyclic_period):
    if args.is_cyclic_precision:
        assert len(args.cyclic_num_bits_schedule) == 2
        assert len(args.cyclic_num_grad_bits_schedule) == 2

        num_bit_min = args.cyclic_num_bits_schedule[0]
        num_bit_max = args.cyclic_num_bits_schedule[1]

        num_grad_bit_min = args.cyclic_num_grad_bits_schedule[0]
        num_grad_bit_max = args.cyclic_num_grad_bits_schedule[1]

        args.num_bits = np.rint(num_bit_min +
                                0.5 * (num_bit_max - num_bit_min) *
                                (1 + np.cos(np.pi * ((_iter % cyclic_period) / cyclic_period) + np.pi)))
        args.num_grad_bits = np.rint(num_grad_bit_min +
                                     0.5 * (num_grad_bit_max - num_grad_bit_min) *
                                     (1 + np.cos(np.pi * ((_iter % cyclic_period) / cyclic_period) + np.pi)))

        if _iter % args.eval_every == 0:
            logging.info('Iter [{}] num_bits = {} num_grad_bits = {} cyclic precision'.format(_iter, args.num_bits,
                                                                                                  args.num_grad_bits))


def adjust_learning_rate(args, optimizer, _iter):
    if args.lr_schedule == 'piecewise':
        if args.warm_up and (_iter < 1):
            lr = 0.01
        elif 50 <= _iter < 150:
            lr = args.lr * (args.step_ratio ** 1)
        elif _iter >= 150:
            lr = args.lr * (args.step_ratio ** 2)
        else:
            lr = args.lr

    elif args.lr_schedule == 'linear':
        t = _iter / args.iters
        lr_ratio = 0.01
        if args.warm_up and (_iter < 400):
            lr = 0.01
        elif t < 0.5:
            lr = args.lr
        elif t < 0.9:
            lr = args.lr * (1 - (1 - lr_ratio) * (t - 0.5) / 0.4)
        else:
            lr = args.lr * lr_ratio

    elif args.lr_schedule == 'anneal_cosine':
        lr_min = args.lr * (args.step_ratio ** 2)
        lr_max = args.lr
        lr = lr_min + 1 / 2 * (lr_max - lr_min) * (1 + np.cos(_iter / args.iters * 3.141592653))

    logging.info('Epoch [{}] learning rate = {}'.format(_iter, lr))
    # if _iter % args.eval_every == 0:
    #     logging.info('Iter [{}] learning rate = {}'.format(_iter, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()