import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from collections import OrderedDict
import os
import logging
from torch.utils.tensorboard import SummaryWriter
from time import process_time
import pdb

from torchmeta.datasets import MiniImagenet, DoubleMNIST, Omniglot
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.transforms import Categorical, ClassSplitter

from gbml.maml_functional_forward import MAML
from gbml.imaml_v2 import iMAML
from gbml.fomaml_ff import FOMAML
from gbml.reptile_final import Reptile
from utils import set_seed, set_gpu, check_dir, dict2tsv, BestTracker, set_logger


def setData(**attributes):
    if args.dataset == 'omniglot':
        return Omniglot(**attributes)
    else:
        return MiniImagenet(**attributes)

def train(args, model, dataloader):

    model.network.train()
    loss_list = []
    acc_list = []
    grad_list = []
    global train_iter
    with tqdm(dataloader, total=args.num_train_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):

            loss_log, acc_log, grad_log = model.outer_loop(
                batch, is_train=True)
            loss_log, acc_log, grad_log = model.outer_loop(
                batch, is_train=True)

            loss_list.append(loss_log)
            acc_list.append(acc_log)
            grad_list.append(grad_log)
            logging.info('step:' + str(batch_idx) +
                         '\tTrain loss:' + repr(round(loss_log, 4)))
            logging.info('step:' + str(batch_idx) +
                         '\tTrain acc:' + repr(round(acc_log, 4)))
            writer.add_scalar('Accuracy/train', round(acc_log, 4), train_iter)
            writer.add_scalar('Loss/train', round(loss_log, 4), train_iter)
            train_iter += 1

            pbar.set_description('loss = {:.4f} || acc={:.4f} || grad={:.4f}'.format(
                np.mean(loss_list), np.mean(acc_list), np.mean(grad_list)))
            if batch_idx >= args.num_train_batches:
                break

    loss = np.round(np.mean(loss_list), 4)
    acc = np.round(np.mean(acc_list), 4)
    grad = np.round(np.mean(grad_list), 4)

    return loss, acc, grad


# @torch.no_grad()
def valid(args, model, dataloader, test):

    model.network.eval()
    global test_iter
    mode = 'valid'
    if test:
        mode = 'test'
    loss_list = []
    acc_list = []
    with tqdm(dataloader, total=args.num_valid_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):

            loss_log, acc_log = model.outer_loop(batch, is_train=False)

            loss_list.append(loss_log)
            acc_list.append(acc_log)
            logging.info('step:' + str(batch_idx) +
                         '\t' + mode + ' loss:' + repr(round(loss_log, 4)))
            logging.info('step:' + str(batch_idx) +
                         '\t' + mode + ' acc:' + repr(round(acc_log, 4)))
            if test:
                writer.add_scalar(
                    'Accuracy/test', round(acc_log, 4), test_iter)
                writer.add_scalar('Loss/test', round(loss_log, 4), test_iter)
                test_iter += 1

            pbar.set_description('loss = {:.4f} || acc={:.4f}'.format(
                np.mean(loss_list), np.mean(acc_list)))
            if batch_idx >= args.num_valid_batches:
                break

    loss = np.round(np.mean(loss_list), 4)
    acc = np.round(np.mean(acc_list), 4)

    return loss, acc


@BestTracker
def run_epoch(epoch, args, model, train_loader, valid_loader, test_loader):

    res = OrderedDict()
    print('Epoch {}'.format(epoch))
    train_loss, train_acc, train_grad = train(args, model, train_loader)
    valid_loss, valid_acc = valid(args, model, valid_loader, False)
    # test_loss, test_acc = valid(args, model, test_loader, True)

    res['epoch'] = epoch
    res['train_loss'] = train_loss
    res['train_acc'] = train_acc
    res['train_grad'] = train_grad
    res['valid_loss'] = valid_loss
    res['valid_acc'] = valid_acc
    # res['test_loss'] = test_loss
    # res['test_acc'] = test_acc

    logging.info('Epoch:' + str(epoch) +
                 '\tTrain loss:' + repr(round(train_loss, 4)))
    logging.info('Epoch:' + str(epoch) +
                 '\tTrain acc:' + repr(round(train_acc, 4)))
    logging.info('Epoch:' + str(epoch) +
                 '\aValid loss:' + repr(round(valid_loss, 4)))
    logging.info('Epoch:' + str(epoch) +
                 '\aValid acc:' + repr(round(valid_acc, 4)))
    # logging.info('Epoch:' + str(epoch) +
    #              '\aTest loss:' + repr(round(test_loss, 4)))
    # logging.info('Epoch:' + str(epoch) +
    #              '\aTest acc:' + repr(round(test_acc, 4)))
    writer.add_scalar('Epoch Accuracy/train', round(train_acc, 4), epoch)
    writer.add_scalar('Epoch Loss/train', round(train_loss, 4), epoch)
    writer.add_scalar('Epoch Accuracy/valid', round(valid_acc, 4), epoch)
    writer.add_scalar('Epoch Loss/valid', round(valid_loss, 4), epoch)
    # writer.add_scalar('Epoch Accuracy/test', round(test_acc, 4), epoch)
    # writer.add_scalar('Epoch Loss/test', round(test_loss, 4), epoch)

    return res


def main(args):
    print(args)
    logging.info(args)

    if args.alg == 'MAML':
        model = MAML(args)
    elif args.alg == 'Reptile':
        model = Reptile(args)
    elif args.alg == 'FOMAML':
        model = FOMAML(args)
    elif args.alg == 'Neumann':
        model = Neumann(args)
    elif args.alg == 'CAVIA':
        model = CAVIA(args)
    elif args.alg == 'iMAML':
        model = iMAML(args)
    else:
        raise ValueError('Not implemented Meta-Learning Algorithm')

    if args.load:
        model.load()

    train_dataset = setData(root=args.data_path, num_classes_per_task=args.num_way,
                            meta_split='train',
                            transform=transforms.Compose([
                                transforms.RandomCrop(80, padding=8),
                                transforms.ColorJitter(
                                    brightness=0.4, contrast=0.4, saturation=0.4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()
                            ]),
                            target_transform=Categorical(
                                num_classes=args.num_way),
                            download=True
                            )
    train_dataset = ClassSplitter(
        train_dataset, shuffle=True, num_train_per_class=args.num_shot, num_test_per_class=args.num_query)
    train_loader = BatchMetaDataLoader(train_dataset, batch_size=args.batch_size,
                                       shuffle=True, pin_memory=True, num_workers=args.num_workers)

    valid_dataset = setData(root=args.data_path, num_classes_per_task=args.num_way,
                            meta_split='val',
                            transform=transforms.Compose([
                                transforms.CenterCrop(80),
                                transforms.ToTensor()
                            ]),
                            target_transform=Categorical(
                                num_classes=args.num_way)
                            )
    valid_dataset = ClassSplitter(
        valid_dataset, shuffle=True, num_train_per_class=args.num_shot, num_test_per_class=args.num_query)
    valid_loader = BatchMetaDataLoader(valid_dataset, batch_size=args.batch_size,
                                       shuffle=True, pin_memory=True, num_workers=args.num_workers)

    test_dataset = setData(root=args.data_path, num_classes_per_task=args.num_way,
                           meta_split='test',
                           transform=transforms.Compose([
                               transforms.CenterCrop(80),
                               transforms.ToTensor()
                           ]),
                           target_transform=Categorical(
                               num_classes=args.num_way)
                           )
    test_dataset = ClassSplitter(
        test_dataset, shuffle=True, num_train_per_class=args.num_shot, num_test_per_class=args.num_query)
    test_loader = BatchMetaDataLoader(test_dataset, batch_size=args.batch_size,
                                      shuffle=True, pin_memory=True, num_workers=args.num_workers)
    global writer
    global train_iter
    global test_iter
    writer = SummaryWriter(comment=args.exp)
    train_iter = 0
    test_iter = 0
    counter = 0
    patience = 15

    for epoch in range(args.num_epoch):

        res, is_best = run_epoch(
            epoch, args, model, train_loader, valid_loader, test_loader)
        dict2tsv(res, os.path.join(args.result_path, args.alg, str(
            args.num_shot) + '_' + str(args.num_way), args.log_path))

        if is_best:
            logging.info('- Found new best accuracy')
            counter = 0  # reset
            model.save()
        else:
            counter += 1

        # disable early stopping
        # if counter > patience:
        #     logging.info('- No improvement in a while, stopping training...')
        #     break

        if args.lr_sched:
            model.lr_sched(res['train_loss'])

    return None


def parse_args():
    import argparse

    parser = argparse.ArgumentParser('Gradient-Based Meta-Learning Algorithms')
    # experimental settings
    parser.add_argument('--seed', type=int, default=2020,
                        help='Random seed.')
    parser.add_argument('--data_path', type=str, default='data/',
                        help='Path of MiniImagenet.')
    parser.add_argument('--result_path', type=str, default='./result')
    parser.add_argument('--log_path', type=str, default='result.tsv')
    parser.add_argument('--save_path', type=str, default='best_model.pth')
    parser.add_argument(
        '--load', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--load_path', type=str, default='best_model.pth')
    parser.add_argument('--device', type=int, nargs='+',
                        default=[0], help='0 = CPU.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading (default: 4).')
    parser.add_argument('--dataset', type=str, default='omniglot')
    # training settings
    parser.add_argument('--num_epoch', type=int, default=400,
                        help='Number of epochs for meta train.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Number of tasks in a mini-batch of tasks (default: 4).')
    parser.add_argument('--num_train_batches', type=int, default=150,
                        help='Number of batches the model is trained over (default: 150).')
    parser.add_argument('--num_valid_batches', type=int, default=150,
                        help='Number of batches the model is trained over (default: 150).')
    # meta-learning settings
    parser.add_argument('--num_shot', type=int, default=1,
                        help='Number of support examples per class (k in "k-shot", default: 1).')
    parser.add_argument('--num_query', type=int, default=15,
                        help='Number of query examples per class (k in "k-query", default: 15).')
    parser.add_argument('--num_way', type=int, default=5,
                        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--alg', type=str, default='iMAML')
    # algorithm settings
    parser.add_argument('--n_inner', type=int, default=5)
    parser.add_argument('--inner_lr', type=float, default=1e-2)
    parser.add_argument('--inner_opt', type=str, default='SGD')
    parser.add_argument('--outer_lr', type=float, default=1e-3)
    parser.add_argument('--outer_opt', type=str, default='Adam')
    parser.add_argument(
        '--lr_sched', type=lambda x: (str(x).lower() == 'true'), default=False)
    # network settings
    parser.add_argument('--net', type=str, default='ConvNet')
    parser.add_argument('--n_conv', type=int, default=4)
    parser.add_argument('--n_dense', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=64,
                        help='Number of channels for each convolutional layer (default: 64).')

    # Added flags
    parser.add_argument('--TTSA', type=str, default='False')
    parser.add_argument('--exp', type=str,
                        help='exp name', default='exp')
    parser.add_argument('--imaml_reg', type=str, default='False')
    parser.add_argument(
        '--n_cg', type=int, help='conjugate gradient steps for inner solver', default=1)
    parser.add_argument('--lamb', type=float,
                        help='regularization factor', default=100)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    set_gpu(args.device)
    check_dir(args)
    set_logger(os.path.join('logs/', args.exp + '.txt'), log_console=False)
    t1_start = process_time()
    main(args)
    t1_stop = process_time()
    logging.info('Elapsed time = {}'.format(t1_stop - t1_start))
