from utils.util import accuracy, AverageMeter, DataPrefetcher, get_logger, seed_torch, warm_up_lr, get_config_file
from utils.select_model import get_network
from utils.select_dataset import get_dataset
from eager_pruner import EagerPruner
from parsers import parse_args

import torch
import os
import torch.backends.cudnn as cudnn
import time
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import models

from thop import profile, clever_format


def train(train_loader, model, criterion, optimizer, scheduler, pruner, epoch, logger,
          args):
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    iters = len(train_loader)
    prefetcher = DataPrefetcher(train_loader)
    #prefetcher = iter(train_loader)
    inputs, labels = prefetcher.next()
    iter_index = 1
    while inputs is not None:
        if epoch <= args.warmup_epoch:
            warm_up_lr(optimizer, epoch, iter_index, args.warmup_epoch, iters, args.lr)
        inputs, labels = inputs.cuda(), labels.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss = loss

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        #剪枝
        if args.use_prune:
            pruner.set_zero_by_mask()
            if pruner.finish_flag == False:
                if pruner.loss_increase_check(loss.item()):
                    if pruner.roll_back_check():
                        logger.info(
                        f"train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iters:0>4d}], roll back")
                        pruner.roll_back(epoch - 1, iter_index - 1)
                        if pruner.pruning_termination_check():
                            pruner.finish_flag = True
                            logger.info(
                                    f"train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iters:0>4d}],{pruner.curr_fail_times:d} times failed, prune finished ")
                        else:
                            pruner.prune_num  = int(pruner.prune_num / 2)
                            if pruner.prune_num == 0:
                                pruner.finish_flag = True
                                logger.info(
                                f"train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iters:0>4d}], prune finished")
                if pruner.pruning_check(epoch - 1, iter_index - 1):
                    logger.info(
                    f"train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iters:0>4d}], begin prune")
                    pruner.backup_and_prune(epoch - 1, iter_index - 1)
                    overall_rate, layers_rate = pruner.get_prune_rate()
                    logger.info(
                    f"train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iters:0>4d}],finish prune, prune_num {pruner.prune_num:d},prune_rate {pruner.current_num/pruner.all_weights_num}，max_smoothed_loss {pruner.last_prune_loss_max:f}")
                    for i, rate in enumerate(layers_rate):
                        logger.info(f"layer {i:0>4d}, rate {rate:.3f}")
            optimizer.zero_grad()
        
        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))
        losses.update(loss.item(), inputs.size(0))


        if iter_index % args.print_interval == 0:
            logger.info(
                f"train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iters:0>4d}], lr: {optimizer.param_groups[0]['lr']:.6f}, top1 acc: {acc1.item():.2f}%, top5 acc: {acc5.item():.2f}%, loss_total: {loss.item():.2f}"
            )

        inputs, labels = prefetcher.next()
        iter_index += 1

    scheduler.step()

    return top1.avg, top5.avg, losses.avg


def validate(val_loader, model, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for inputs, labels in val_loader:
            data_time.update(time.time() - end)
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

    throughput = 1.0 / (batch_time.avg / inputs.size(0))

    return top1.avg, top5.avg, throughput


def main(logger, args):
    if not torch.cuda.is_available():
        raise Exception("need gpu to train network!")
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(g) for g in args.gpus])
    print(os.environ["CUDA_VISIBLE_DEVICES"])

    if args.seed is not None:
        seed_torch(args.seed)

    logger.info(f'use {args.gpus} gpu')
    logger.info(f"args: {args}")

    cudnn.benchmark = True
    cudnn.enabled = True
    start_time = time.time()

    # dataset and dataloader
    logger.info('start loading data')
    train_dataset, val_dataset = get_dataset(args)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=args.num_workers)
    logger.info('finish loading data')

    logger.info(f"creating model '{args.net}'")
    model = get_network(args)

    if args.dataset == 'mnist':
        flops_input = torch.randn(1, 1, args.input_image_size,
                              args.input_image_size)
    else:
        flops_input = torch.randn(1, 3, args.input_image_size,
                              args.input_image_size)
    flops, params = profile(model, inputs=(flops_input, ))
    flops, params = clever_format([flops, params], "%.3f")
    logger.info(f"model: '{args.net}', flops: {flops}, params: {params}")

    for name, param in model.named_parameters():
        logger.info(f"{name},{param.requires_grad}")

    #model = model.to(f'cuda:{args.gpus[0]}')
    #criterion = nn.CrossEntropyLoss().to(f'cuda:{args.gpus[0]}')
    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.gamma)
    
    #model = nn.DataParallel(model, device_ids=args.gpus)
    model = nn.DataParallel(model)
    
    if args.use_prune:
        pruner = EagerPruner(model, args.epochs, len(train_loader), 
                            prune_interval=args.prune_interval,
                            prune_num=args.prune_num,
                            over_prune_threshold=args.over_prune_threshold,
                            prune_fail_times=args.prune_fail_times)
        logger.info(
                f"all weights in conv layers {pruner.all_weights_num:d}")
    else:
        pruner = None

    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            raise Exception(
                f"{args.resume} is not a file, please check it again")
        logger.info('start only evaluating')
        logger.info(f"start resuming model from {args.evaluate}")
        checkpoint = torch.load(args.evaluate,
                                map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        acc1, acc5, throughput = validate(val_loader, model, args)
        logger.info(
            f"epoch {checkpoint['epoch']:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: {acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )

        return

    start_epoch = 1
    # resume training
    if os.path.exists(args.resume):
        logger.info(f"start resuming model from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        start_epoch += checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(
            f"finish resuming model from {args.resume}, epoch {checkpoint['epoch']}, "
            f"loss: {checkpoint['loss']:3f}, lr: {checkpoint['lr']:.6f}, "
            f"top1_acc: {checkpoint['acc1']}%")

    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)

    logger.info('start training')
    for epoch in range(start_epoch, args.epochs + 1):
        acc1, acc5, losses = train(train_loader, model, criterion, optimizer,
                                   scheduler, pruner, epoch, logger, args)
        logger.info(
            f"train: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: {acc5:.2f}%, losses: {losses:.2f}"
        )

        acc1, acc5, throughput = validate(val_loader, model, args)
        logger.info(
            f"val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: {acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )

        # remember best prec@1 and save checkpoint
        torch.save(
            {
                'epoch': epoch,
                'acc1': acc1,
                'loss': losses,
                'lr': optimizer.param_groups[0]['lr'],
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(args.checkpoints, 'latest.pth'))
        if epoch == args.epochs:
            torch.save(
                model.module.state_dict(),
                os.path.join(
                    args.checkpoints,
                    "{}-epoch{}-acc{}.pth".format(args.network, epoch, acc1)))

    training_time = (time.time() - start_time) / 3600
    logger.info(
        f"finish training, total training time: {training_time:.2f} hours")
    
    if args.use_prune:
        overall_rate, layers_rate = pruner.get_prune_rate()        
        logger.info(
            f"finish training, total prune rate: {overall_rate:.3f}")
        for i, rate in enumerate(layers_rate):
            logger.info(f"layer {i:0>4d}, rate {rate:.3f}")

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(g) for g in args.gpus])
    print(args.__dict__)
    get_config_file(args)
    print(args.__dict__)
    logger = get_logger(__name__, args.log)
    main(logger, args)
