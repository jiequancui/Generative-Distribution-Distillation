"""
the general training framework
"""

from __future__ import print_function

import os
import argparse
import socket
import time
import numpy as np
import math

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn


from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample

from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss
from crd.criterion import CRDLoss

from helper.loops import train_distill as train, validate
from helper.pretrain import init


def param_groups_setup(model, opt, skip_list=()):
    diff_params, non_diff_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  
        # group setup
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list or 'diffloss' in name:
            diff_params.append(param)
        else:
            non_diff_params.append(param)

    return [
        {'params': diff_params, 'weight_decay': opt.diff_weight_decay},
        {'params': non_diff_params, 'weight_decay': opt.weight_decay}]



def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # optimization
    parser.add_argument('--warmup_epochs', type=int, default=10, help='warmup epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--min_learning_rate', type=float, default=1e-6, help='minimal learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list') #150 180 210
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'wrn_34_10'])
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    # evaluation
    parser.add_argument('--resume', type=str, default=None, help='resume checkpoint')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--finetune_classifier', action='store_true')
    parser.add_argument('--smooth', type=float, default=0.9, help='smooth technique')
    parser.add_argument('--diff_weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--diffloss_w', type=int, default=1024, help='hidden width of diffusion model')
    parser.add_argument('--adam_beta', type=float, default=0.95, help='beta for adamw')
    parser.add_argument('--diffusion_batch_mul', type=int, default=50, help='beta for adamw')
    parser.add_argument('--cos', action='store_true')


    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/student_model'
        opt.tb_path = '/path/to/my/student_tensorboards'
    else:
        opt.model_path = './save/student_model'
        opt.tb_path = './save/student_tensorboards'

    if opt.epochs == 480:
        opt.lr_decay_epochs = '300,360,420'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)

    opt.model_name = 'S:{}_T:{}_{}_stage2_{}'.format(opt.model_s, opt.model_t, opt.dataset, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]

def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    state_dict = torch.load(model_path)['model']
    new_state_dict = {}
    for key in state_dict.keys():
        if key == 'classifier.weight' or key == 'linear.weight':
            new_state_dict['fc.weight'] = state_dict[key]
        elif key == 'classifier.bias' or key == 'linear.bias':
            new_state_dict['fc.bias'] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]

    model.load_state_dict(new_state_dict)
    print('==> done')
    return model

def adjust_learning_rate_cos(epoch, opt, optimizer):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < opt.warmup_epochs:
        lr = opt.learning_rate * epoch / opt.warmup_epochs 
    else:
        lr = opt.min_learning_rate + (opt.learning_rate - opt.min_learning_rate) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - opt.warmup_epochs) / (opt.epochs - opt.warmup_epochs)))
    for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    return lr

def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""

    if epoch < opt.warmup_epochs:
        new_lr = opt.learning_rate * epoch / opt.warmup_epochs 
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

def main():
    best_acc = 0

    opt = parse_option()

    # tensorboard logger
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers, is_instance=True)
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model_t = load_teacher(opt.path_t, n_cls)
    opt.target_dim = model_t.fc.weight.size(1)
    if hasattr(model_t, 'fc'):
        classifier = model_t.fc
    elif hasattr(model_t, 'linear'):
        classifier = model_t.linear

    model = model_dict[opt.model_s](num_classes=n_cls)
    opt.feature_dim = model.fc.weight.size(1)
    model_s = model_dict['DiffModel'](model=model, classifier=classifier, feature_dim=opt.feature_dim, target_dim=opt.target_dim, finetune_classifier=opt.finetune_classifier, stage='stage-2', smooth=opt.smooth, diffloss_w=opt.diffloss_w, diffusion_batch_mul=opt.diffusion_batch_mul)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss

    # optimizer
    param_groups = param_groups_setup(trainable_list, opt)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, opt.adam_beta), lr=opt.learning_rate)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    if opt.resume is not None:
        checkpoint = torch.load(opt.resume)['model']
        model_s.load_state_dict(checkpoint, strict=False)

    if opt.evaluate:
        assert opt.resume is not None, " model path should be provided before evaluation"
        acc, _, _ = validate(val_loader, model_s, criterion_cls, opt)
        print('model accuracy: ', acc)
        return

    # validate teacher accuracy
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: ', teacher_acc)

    # routine
    for epoch in range(1, opt.epochs + 1):

        if opt.cos:
            adjust_learning_rate_cos(epoch, opt, optimizer)
        else:
            adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, tect_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        logger.log_value('test_acc_top5', tect_acc_top5, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'accuracy': test_acc,
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch. 
    print('best accuracy:', best_acc)

    # save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
