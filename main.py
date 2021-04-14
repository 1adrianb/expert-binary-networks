import os
import random
import time
import warnings
import json

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from warmup_scheduler import GradualWarmupScheduler
from models import resnet_generic
from models.eb_resnet import EBBasicBlock, EBDeepBasicBlock
from bnn import BConfig, prepare_binary_model
from bnn.ops import BasicInputBinarizer, BasicScaleBinarizer, XNORWeightBinarizer
from utils.mixup import mixup_criterion, mixup_data
from utils.distillation_losses import LogitMatch, AttentionMatching
from utils.misc import *

from models.ebconv import EBConv2d
from opts import parser

best_acc1 = 0


def main():
    args = parser.parse_args()

    args_dict = vars(args)
    print(args_dict)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    with open(f'{args.output_dir}/args.txt', 'w') as fd:
        json.dump(args_dict, fd, indent=4)

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(
            main_worker,
            nprocs=ngpus_per_node,
            args=(
                ngpus_per_node,
                args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank)

    num_classes = 1000

    ignore_layers_name = [
        'conv1',
        'fc',
        '$layer+[0-9]\.0\.downsample\.+[0-9]$']

    # create model
    print('=> creating model ...')
    model = resnet_generic(
        block_type=EBDeepBasicBlock if args.add_g_layer else EBBasicBlock,
        structure=args.structure,
        groups=args.num_groups,
        expansion=args.expansion,
        stem_type=args.stem_type,
        num_classes=num_classes,
        activation=nn.PReLU,
        num_experts=args.num_experts,
        use_only_first=args.use_only_first,
        use_se=args.use_se,
        downsample_ratio=args.downsample_ratio
    )
    bconfig = BConfig(
        activation_pre_process=BasicInputBinarizer if args.binary_activations else nn.Identity,
        activation_post_process=BasicScaleBinarizer,
        weight_pre_process=XNORWeightBinarizer.with_args(
            compute_alpha=False) if args.binary_weights else nn.Identity)
    model = prepare_binary_model(
        model,
        bconfig=bconfig,
        modules_mapping={
            EBConv2d: EBConv2d},
        ignore_layers_name=ignore_layers_name)
    print(model)

    print(f'Num paramters: {count_parameters(model)}')

    # Load teacher config if needed
    if args.teacher_config != '':
        with open(args.teacher_config, 'r') as fd:
            teacher_args = json.load(fd)
    teacher = None
    if args.teacher != '':
        print('=> creating teacher model ')
        teacher = resnet_generic(
            block_type=EBDeepBasicBlock if teacher_args['add_g_layer'] else EBBasicBlock,
            structure=teacher_args['structure'],
            groups=teacher_args['num_groups'],
            expansion=teacher_args['expansion'],
            stem_type=teacher_args['stem_type'],
            num_classes=num_classes,
            activation=nn.PReLU,
            num_experts=teacher_args['num_experts'],
            use_only_first=teacher_args['use_only_first'],
            use_se=teacher_args['use_se'],
            downsample_ratio=teacher_args['downsample_ratio'])
        bconfig = BConfig(
            activation_pre_process=BasicInputBinarizer if teacher_args['binary_activations'] else nn.Identity,
            activation_post_process=BasicScaleBinarizer,
            weight_pre_process=XNORWeightBinarizer.with_args(
                compute_alpha=False) if teacher_args['binary_weights'] else nn.Identity)
        teacher = prepare_binary_model(
            teacher,
            bconfig=bconfig,
            modules_mapping={
                EBConv2d: EBConv2d},
            ignore_layers_name=ignore_layers_name)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            if teacher is not None:
                teacher.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(
                (args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
            if teacher is not None:
                teacher = torch.nn.parallel.DistributedDataParallel(
                    teacher, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            if teacher is not None:
                teacher.cuda()
                teacher = torch.nn.parallel.DistributedDataParallel(teacher)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        if teacher is not None:
            teacher = teacher.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available
        # GPUs
        model = torch.nn.DataParallel(model).cuda()
        if teacher is not None:
            teacher = torch.nn.DataParallel(teacher).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    criterion_kd = LogitMatch(
        T=args.lab_match_T,
        weight=args.lab_match_w) if args.lab_match else None
    criterion_att = AttentionMatching(
        args.att_transfer_weighting,
        args.att_transfer_indicator) if args.att_transfer else None

    parameters = model.parameters()
    if args.optimizer == 'adamw':
        wd = 0 if args.binary_weights else args.weight_decay
        optimizer = torch.optim.AdamW(parameters, args.lr, weight_decay=wd)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(parameters, args.lr)
    elif args.optimizer == 'sgd':
        wd = 0 if args.binary_weights else args.weight_decay
        optimizer = torch.optim.SGD(parameters, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=wd)
    else:
        raise ValueError(f'Unknown optimizer selected: {args.optimizer}')

    if args.scheduler == 'multistep':
        milestone = [40, 70, 80, 100, 110]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[x - args.warmup for x in milestone], gamma=0.1)
    elif args.scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(args.epochs - args.warmup), eta_min=0)
    else:
        raise ValueError(f'Unknown schduler selected: {args.scheduler}')

    if args.warmup > 0:
        print(f'=> Applying warmup ({args.warmup} epochs)')
        lr_scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=args.warmup,
            after_scheduler=lr_scheduler)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            if args.resume_epoch:
                args.start_epoch = checkpoint['epoch']
                best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                pass
                # best_acc1 may be from a checkpoint from a different GPU
                #best_acc1 = best_acc1.to(args.gpu)
            try:
                model.load_state_dict(checkpoint['state_dict'])
                if not ('adam' in args.optimizer and 'sgd' in args.resume):
                    print('=> Loading optimizer...')
                    # optimizer.load_state_dict(checkpoint['optimizer'])
            except BaseException:
                print('=> Warning: dict model mismatch, loading with strict = False')
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

        # Reset learning rate
        for g in optimizer.param_groups:
            g['lr'] = args.lr

        if args.expansion_stage:
            print('Expanding the weights...')
            for module in model.modules():
                if isinstance(module, EBConv2d):
                    if not isinstance(
                            module.activation_pre_process,
                            nn.Identity):
                        print(
                            f'Init module with w shape = {module.weight.size()}')
                        for i in range(1, args.num_experts):
                            module.weight.data[i, ...].copy_(
                                module.weight.data[0, ...])

    if args.start_epoch > 0:
        print(f'Advancing the scheduler to epoch {args.start_epoch}')
        for i in range(args.start_epoch):
            lr_scheduler.step()
    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'valid')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transforms_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms_train)
    val_dataset = datasets.ImageFolder(valdir, transforms_val)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(
            train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    show_logs = (
        not args.multiprocessing_distributed) or (
        args.multiprocessing_distributed and args.rank %
        ngpus_per_node == 0)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if args.scheduler == 'cosine':
            lr_scheduler.step(epoch)
        else:
            lr_scheduler.step()
        if show_logs:
            print(f'New lr: {lr_scheduler.get_last_lr()}')

        # train for one epoch
        train(
            train_loader,
            model,
            teacher,
            criterion,
            optimizer,
            epoch,
            args,
            criterion_kd=criterion_kd,
            criterion_att=criterion_att,
            show_logs=show_logs)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args, show_logs)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        print(f'Current best: {best_acc1}')

        if show_logs:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.output_dir)


def train(
        train_loader,
        model,
        teacher,
        criterion,
        optimizer,
        epoch,
        args,
        criterion_kd=None,
        criterion_att=None,
        show_logs=True):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    all_meters = [batch_time, data_time, losses, top1, top5]
    if criterion_kd is not None:
        losses_kd = AverageMeter('Loss KD', ':.4e')
        all_meters.append(losses_kd)
    if criterion_att is not None:
        losses_att = AverageMeter('Loss Att', ':.4e')
        all_meters.append(losses_att)

    progress = ProgressMeter(
        len(train_loader),
        all_meters,
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        if args.use_mixup:
            images, target_a, target_b, lam = mixup_data(
                images, target, args.alpha)

        if teacher is not None:
            with torch.no_grad():
                output_teacher, teacher_interim = teacher(images)
        # compute output
        output, interim = model(images)

        loss = mixup_criterion(
            criterion,
            output,
            target_a,
            target_b,
            lam) if args.use_mixup else criterion(
            output,
            target)

        loss_att = criterion_att(
            interim, teacher_interim) if criterion_att is not None else 0
        loss += loss_att

        loss_kd = criterion_kd(
            output_s=output,
            output_t=output_teacher) if criterion_kd is not None else 0
        loss += loss_kd

        # measure accuracy and record loss
        if args.use_mixup:
            acc1a, acc5a = accuracy(output, target_a, topk=(1, 5))
            acc1b, acc5b = accuracy(output, target_b, topk=(1, 5))
            acc1 = lam * acc1a + (1 - lam) * acc1b
            acc5 = lam * acc5a + (1 - lam) * acc5b
        else:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        if criterion_kd is not None:
            losses_kd.update(loss_kd.item(), images.size(0))
        if criterion_att is not None:
            losses_att.update(loss_att.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and show_logs:
            progress.display(i)


def validate(val_loader, model, criterion, args, show_logs=True):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            output, _ = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and show_logs:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        if show_logs:
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))

    return top1.avg


if __name__ == '__main__':
    main()
