# -*- coding: utf-8 -*-
"""
Reference:
@article{goldblum2020AQ,
  title={Adversarially Robust Few-Shot Learning: A Meta-Learning Approach},
  author={Goldblum, Micah and Fowl, Liam and Goldstein, Tom},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}

"""
import os
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import pandas as pd

from models.classification_heads import ClassificationHead
from models.R2D2_embedding import R2D2Embedding
from models.protonet_embedding import ProtoNetEmbedding
from models.ResNet12_embedding import resnet12

from utils import set_gpu, Timer, count_accuracy, check_dir, log, AttackPGD, draw_figure


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.

    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)

    return encoded_indicies


def get_model(options, device):
    # Choose the embedding network
    if options.model == 'ProtoNet':
        model = ProtoNetEmbedding(activation=opt.activation, whether_denoising=opt.whether_denoising,
                                  filter_type=opt.filter_type).to(device)
    elif options.model == 'R2D2':
        model = R2D2Embedding(denoise=opt.denoise, activation=opt.activation, whether_denoising=opt.whether_denoising,
                              filter_type=opt.filter_type).to(device)
    elif options.model == 'ResNet':
        if options.dataset == 'miniImageNet' or opt.dataset == 'tieredImageNet':
            model = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=5, whether_denoising=opt.whether_denoising,
                             filter_type=opt.filter_type).to(device)
            # model = torch.nn.DataParallel(model, device_ids=[1])
        else:
            model = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=2, whether_denoising=opt.whether_denoising,
                             filter_type=opt.filter_type).to(device)

    # Choose the classification head
    if options.head == 'ProtoNet':
        cls_head = ClassificationHead(base_learner='ProtoNet').to(device)
    elif options.head == 'Ridge':
        cls_head = ClassificationHead(base_learner='Ridge').to(device)
    elif options.head == 'R2D2':
        cls_head = ClassificationHead(base_learner='R2D2').to(device)
    elif options.head == 'SVM':
        cls_head = ClassificationHead(base_learner='SVM-CS').to(device)
    else:
        print("Cannot recognize the dataset type")
        assert False

    return model, cls_head


def get_dataset(options):
    # Choose the embedding network
    if options.dataset == 'miniImageNet':
        from data.mini_imagenet import MiniImageNet, FewShotDataloader
        dataset_train = MiniImageNet(phase='train')
        dataset_val = MiniImageNet(phase='val')
        data_loader = FewShotDataloader
    elif options.dataset == 'tieredImageNet':
        from data.tiered_imagenet import tieredImageNet, FewShotDataloader
        dataset_train = tieredImageNet(phase='train')
        dataset_val = tieredImageNet(phase='val')
        data_loader = FewShotDataloader
    elif options.dataset == 'CIFAR_FS':
        from data.CIFAR_FS import CIFAR_FS, FewShotDataloader
        dataset_train = CIFAR_FS(phase='train')
        dataset_val = CIFAR_FS(phase='val')
        data_loader = FewShotDataloader
    elif options.dataset == 'FC100':
        from data.FC100 import FC100, FewShotDataloader
        dataset_train = FC100(phase='train')
        dataset_val = FC100(phase='val')
        data_loader = FewShotDataloader
    else:
        print("Cannot recognize the dataset type")
        assert False

    return dataset_train, dataset_val, data_loader


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1, 6, 7"
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epoch', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='frequency of model saving')
    parser.add_argument('--train-shot', type=int, default=15,
                        help='number of support examples per training class')
    parser.add_argument('--val-shot', type=int, default=1,
                        help='number of support examples per validation class')
    parser.add_argument('--train-query', type=int, default=6,
                        help='number of query examples per training class')
    parser.add_argument('--val-episode', type=int, default=2000,
                        help='number of episodes per validation')
    parser.add_argument('--val-query', type=int, default=15,
                        help='number of query examples per validation class')
    parser.add_argument('--train-way', type=int, default=5,
                        help='number of classes in one training episode')
    parser.add_argument('--test-way', type=int, default=5,
                        help='number of classes in one test (or validation) episode')
    parser.add_argument('--dataset', type=str, default='FC100',
                        help='choose which classification head to use. miniImageNet, tieredImageNet, CIFAR_FS, FC100')
    parser.add_argument('--gpu', default='1, 6, 7')

    parser.add_argument('--head', type=str, default='R2D2',
                        help='choose which classification head to use for model. ProtoNet, Ridge, R2D2, SVM')
    parser.add_argument('--model', default='ProtoNet', type=str, help='ProtoNet, R2D2, ResNet')

    parser.add_argument('--episodes-per-batch', type=int, default=8,
                        help='number of episodes per batch')
    parser.add_argument('--eps', type=float, default=0.0,
                        help='epsilon of label smoothing')
    parser.add_argument('--attack_embedding', action='store_false',
                        help='use attacks to train embedding? store_false to AT')
    parser.add_argument('--attack_epsilon', type=float, default=8.0 / 255.0,
                        help='epsilon for linfinity ball in which images are perturbed')
    parser.add_argument('--attack_steps', type=int, default=7,
                        help='number of PGD steps for each attack')
    parser.add_argument('--attack_step_size', type=float, default=2.0 / 255.0,
                        help='number of query examples per training class')
    parser.add_argument('--attack_targeted', action='store_true',
                        help='used targeted attacks')
    parser.add_argument('--denoise', action='store_true',
                        help='use feature denoising')
    parser.add_argument('--lr', type=float, default=1e-03,
                        help='lr')
    parser.add_argument('--lr_decay', type=float, default=1e-6,
                        help='lr_decay')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='wd')
    parser.add_argument('--TRADES_coef', type=float, default=1.0,
                        help='coef for second term of trades loss')
    parser.add_argument('--activation', type=str, default='LeakyReLU',
                        help='choose which activation function to use. only implemented for R2D2 and ProtoNet')
    parser.add_argument('--checkpoint_epoch', default=0, type=int, help='epoch for loading teacher model')
    parser.add_argument('--meta_attack', default='Query',
                        help='Query for attack query set only, Query_Support for attack query/support both')
    parser.add_argument('--whether_denoising', default=True,
                        help='whether import denoise block')
    parser.add_argument('--filter_type', default='Gaussian_Filter',
                        help='denoising filter, Median_Filter, Mean_Filter, Gaussian_Filter')       

    opt = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_train, dataset_val, data_loader = get_dataset(opt)

    config = {
        'epsilon': opt.attack_epsilon,
        'num_steps': opt.attack_steps,
        'step_size': opt.attack_step_size,
        'targeted': opt.attack_targeted,
        'random_init': True
    }

    # Dataloader of Gidaris & Komodakis (CVPR 2018)
    dloader_train = data_loader(
        dataset=dataset_train,
        nKnovel=opt.train_way,
        nKbase=0,
        nExemplars=opt.train_shot,  # num training examples per novel category
        nTestNovel=opt.train_way * opt.train_query,  # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=opt.episodes_per_batch,
        num_workers=4,
        epoch_size=opt.episodes_per_batch * 1000,  # num of batches per epoch
    )

    dloader_val = data_loader(
        dataset=dataset_val,
        nKnovel=opt.test_way,
        nKbase=0,
        nExemplars=opt.val_shot,  # num training examples per novel category
        nTestNovel=opt.val_query * opt.test_way,  # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * opt.val_episode,  # num of batches per epoch
    )

    set_gpu(opt.gpu)
    checkpoint_path = './experiments/{}/{}/{}/AQ_TRADE'.format(opt.meta_attack, opt.dataset, opt.model)
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path + '/', )

    print('==> Building teacher model and cls head')
    model, cls_head = get_model(opt, device)
    for param in model.parameters():
        requires_grad = True
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    log_file_path = os.path.join(checkpoint_path, "teacher_train_log.txt")
    avg_loss_log_file_path = os.path.join(checkpoint_path, "avg_loss_log.csv")
    loss_log_file_path = os.path.join(checkpoint_path, "loss_log.csv")
    avg_acc_log_file_path = os.path.join(checkpoint_path, "avg_acc_log.csv")
    acc_log_file_path = os.path.join(checkpoint_path, "acc_log.csv")
    avg_val_loss_log_file_path = os.path.join(checkpoint_path, "avg_val_loss_log.csv")
    avg_val_acc_log_file_path = os.path.join(checkpoint_path, "avg_val_acc_log.csv")
    log(log_file_path, str(vars(opt)))

    lambda_epoch = lambda e: 1.0 if e < 20 else (0.06 if e < 40 else 0.012 if e < 50 else 0.0024)

    optimizer = torch.optim.Adam([{'params': model.parameters()},
                                  {'params': cls_head.parameters()}], lr=0.1, weight_decay=5e-4)

    # optimizer = torch.optim.SGD([{'params': model.parameters()},
    #                              {'params': cls_head.parameters()}], lr=0.1, momentum=0.9,
    #                             weight_decay=5e-4, nesterov=True)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)

    max_val_acc = 0.0
    timer = Timer()
    x_entropy = torch.nn.CrossEntropyLoss()
    # (embedding_net, cls_head) = get_model(opt)
    for epoch in range(1, opt.num_epoch + 1):
        # Train on the training split
        # AT for every 10 epoch
        attack_embedding = opt.attack_embedding
        sub_epoch = epoch % 10
        if sub_epoch in [6, 7, 8, 9, 0]:
            attack_embedding = True
        lr_scheduler.step()
        # Fetch the current epoch's learning rate
        epoch_learning_rate = 0.1
        for param_group in optimizer.param_groups:
            epoch_learning_rate = param_group['lr']

        log(log_file_path, 'Train Epoch: {}\tLearning Rate: {:.4f}'.format(
            epoch, epoch_learning_rate))

        _, _ = [x.train() for x in (model, cls_head)]

        t_train_accuracies = []
        t_train_losses = []
        for i, batch in enumerate(tqdm(dloader_train(epoch)), 1):
            data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]

            train_n_support = opt.train_way * opt.train_shot
            train_n_query = opt.train_way * opt.train_query

            emb_support = model(data_support.reshape([-1] + list(data_support.shape[-3:])))
            emb_support = emb_support.reshape(opt.episodes_per_batch, train_n_support, -1)
            data_query = AttackPGD(attack_embedding, model, cls_head, config, data_query,
                                   emb_support, labels_query, labels_support, opt.train_way,
                                   opt.train_shot, opt.head, opt.episodes_per_batch, train_n_query)

            emb_query = model(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(opt.episodes_per_batch, train_n_query, -1)
            if opt.meta_attack != 'Query':
                data_support_adv = AttackPGD(attack_embedding, model, cls_head, config, data_support,
                                             emb_support, labels_support, labels_support, opt.train_way,
                                             opt.train_shot, opt.head, opt.episodes_per_batch, train_n_support)

                emb_support_adv = model(data_support_adv.reshape([-1] + list(data_support.shape[-3:])))
                emb_support_adv = emb_support_adv.reshape(opt.episodes_per_batch, train_n_support, -1)

                logit_query = cls_head(emb_query, emb_support_adv, labels_support,
                                       opt.train_way, opt.train_shot)
            else:
                logit_query = cls_head(emb_query, emb_support, labels_support,
                                       opt.train_way, opt.train_shot)

            smoothed_one_hot_model = one_hot(labels_query.reshape(-1), opt.train_way)
            smoothed_one_hot_model = smoothed_one_hot_model * (1 - opt.eps) + (
                    1 - smoothed_one_hot_model) * opt.eps / (opt.train_way - 1)

            if opt.TRADES_coef > 0.0 and sub_epoch in [6, 7, 8, 9, 0]:

                kl_criterion = torch.nn.KLDivLoss(reduction='batchmean')
                data_query_clean = AttackPGD(False, model, cls_head, config, data_query, emb_support,
                                             labels_query, labels_support, opt.train_way, opt.train_shot,
                                             opt.head, opt.episodes_per_batch, train_n_query)
                emb_query_clean = model(data_query_clean.reshape([-1] + list(data_query.shape[-3:])))
                emb_query_clean = emb_query_clean.reshape(opt.episodes_per_batch, train_n_query, -1)
                logit_query_clean = cls_head(emb_query_clean, emb_support, labels_support, opt.train_way,
                                             opt.train_shot)
                loss_kl = kl_criterion(F.log_softmax(logit_query, dim=1), F.softmax(logit_query_clean, dim=1))

                log_prb = F.log_softmax(logit_query.reshape(-1, opt.train_way), dim=1)
                loss_xent = -(smoothed_one_hot_model * log_prb).sum(dim=1)
                loss_xent = loss_xent.mean()
                loss = loss_xent + opt.TRADES_coef * loss_kl

            else:
                logit_query = cls_head(emb_query, emb_support, labels_support,
                                       opt.train_way, opt.train_shot)
                log_prb = F.log_softmax(logit_query.reshape(-1, opt.train_way), dim=1)
                loss_xent = -(smoothed_one_hot_model * log_prb).sum(dim=1)
                loss_xent = loss_xent.mean()
                loss = loss_xent

            t_acc = count_accuracy(logit_query.reshape(-1, opt.train_way), labels_query.reshape(-1))

            t_train_accuracies.append(t_acc.item())
            t_train_losses.append(loss.item())
            if i % 100 == 0:
                t_train_acc_avg = np.mean(np.array(t_train_accuracies))
                log(log_file_path,
                    'Train Epoch: {}\tBatch: [{}/{}]\tLoss: {:.4f}\tAccuracy: {:.2f} % ({:.2f} %)'.format(
                        epoch, i, len(dloader_train), loss.item(), t_train_acc_avg, t_acc))
                log(loss_log_file_path, '{},{:.4f}'.format(i + (epoch - 1) * 1000, loss.item()))
                log(acc_log_file_path, '{},{:.4f}'.format(i + (epoch - 1) * 1000, t_train_acc_avg))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate on the validation split
        _, _ = [x.eval() for x in (model, cls_head)]

        acc = []
        loss = []
        for i, batch in enumerate(tqdm(dloader_val(epoch)), 1):
            data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]

            test_n_support = opt.test_way * opt.val_shot
            test_n_query = opt.test_way * opt.val_query

            emb_support = model(data_support.reshape([-1] + list(data_support.shape[-3:])))
            emb_support = emb_support.reshape(1, test_n_support, -1)

            data_query = AttackPGD(attack_embedding, model, cls_head, config, data_query,
                                   emb_support, labels_query, labels_support, opt.test_way,
                                   opt.val_shot, opt.head, 1, test_n_query)

            emb_query = model(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(1, test_n_query, -1)

            logit_query = cls_head(emb_query, emb_support, labels_support,
                                   opt.test_way, opt.val_shot)

            t_loss = x_entropy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))
            t_acc = count_accuracy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))
            acc.append(t_acc.item())
            loss.append(t_loss.item())

        t_val_acc_avg = np.mean(np.array(acc))
        t_val_acc_ci95 = 1.96 * np.std(np.array(acc)) / np.sqrt(opt.val_episode)
        t_val_loss_avg = np.mean(np.array(loss))
        if t_val_acc_avg > max_val_acc:
            max_val_acc = t_val_acc_avg
            torch.save({'embedding': model.state_dict(), 'head': cls_head.state_dict()},
                       os.path.join(checkpoint_path, 'best_model.pth'))
            log(log_file_path,
                '\n Teacher Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} % (Best)'
                .format(epoch, t_val_loss_avg, t_val_acc_avg, t_val_acc_ci95))
        else:
            log(log_file_path,
                '\n Teacher Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} %'
                .format(epoch, t_val_loss_avg, t_val_acc_avg, t_val_acc_ci95))

        torch.save({'embedding': model.state_dict(), 'head': cls_head.state_dict()}
                   , os.path.join(checkpoint_path, 'last_teacher_epoch.pth'))

        if epoch % opt.save_epoch == 0:
            torch.save({'embedding': model.state_dict(), 'head': cls_head.state_dict()}
                       , os.path.join(checkpoint_path,
                                      'dataset_{}_model_{}_epoch_{}.pth'.format(opt.dataset, opt.model, epoch)))

        log(log_file_path,
            'Elapsed Time: {}/{}\n'.format(timer.measure(), timer.measure(epoch / float(opt.num_epoch))))

        log(avg_val_loss_log_file_path, '{},{:.4f}'.format(epoch, t_val_loss_avg))
        log(avg_val_acc_log_file_path, '{},{:.4f}'.format(epoch, t_val_acc_avg))
        log(avg_acc_log_file_path, '{},{:.4f}'.format(epoch, np.mean(np.array(t_train_accuracies))))
        log(avg_loss_log_file_path, '{},{:.4f}'.format(epoch, np.mean(np.array(t_train_losses))))

    draw_figure(checkpoint_path + '/')
