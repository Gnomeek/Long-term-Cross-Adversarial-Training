# -*- coding: utf-8 -*-
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from models.protonet_embedding import ProtoNetEmbedding
from models.R2D2_embedding import R2D2Embedding
from models.ResNet12_embedding import resnet12
from models.classification_heads import ClassificationHead
from utils import pprint, set_gpu, Timer, count_accuracy, log, AttackPGD
import numpy as np
import os


def get_dataset(options):
    # Choose the embedding network
    if options.dataset == 'miniImageNet':
        from data.mini_imagenet import MiniImageNet, FewShotDataloader
        dataset_test = MiniImageNet(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'tieredImageNet':
        from data.tiered_imagenet import tieredImageNet, FewShotDataloader
        dataset_test = tieredImageNet(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'CIFAR_FS':
        from data.CIFAR_FS import CIFAR_FS, FewShotDataloader
        dataset_test = CIFAR_FS(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'FC100':
        from data.FC100 import FC100, FewShotDataloader
        dataset_test = FC100(phase='test')
        data_loader = FewShotDataloader
    else:
        print("Cannot recognize the dataset type")
        assert False

    return (dataset_test, data_loader)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--episode', type=int, default=1000,
                        help='number of episodes to test')
    parser.add_argument('--way', type=int, default=5,
                        help='number of classes in one test episode')
    parser.add_argument('--shot', type=int, default=1,
                        help='number of support examples per training class')
    parser.add_argument('--query', type=int, default=15,
                        help='number of query examples per training class')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                        help='choose which classification head to use. miniImageNet, tieredImageNet, CIFAR_FS, FC100')

    parser.add_argument('--model', default='ProtoNet', type=str,
                        help='ProtoNet, R2D2, ResNet')
    parser.add_argument('--head', type=str, default='R2D2',
                        help='choose which classification head to use for teacher model. ProtoNet, Ridge, R2D2, SVM')

    parser.add_argument('--attack_embedding', action='store_false',
                        help='use attacks to train embedding? store_false to AT')
    parser.add_argument('--attack_epsilon', type=float, default=8.0 / 255.0,
                        help='epsilon for linfinity ball in which images are perturbed')
    parser.add_argument('--attack_steps', type=int, default=20,
                        help='number of PGD steps for each attack')
    parser.add_argument('--attack_step_size', type=float, default=2.0 / 255.0,
                        help='number of query examples per training class')
    parser.add_argument('--attack_targeted', action='store_true',
                        help='used targeted attacks')

    parser.add_argument('--activation', type=str, default='LeakyReLU',
                        help='choose which activation function to use. only implemented for R2D2 and ProtoNet')
    parser.add_argument('--denoise', action='store_true',
                        help='use feature denoising')
    parser.add_argument('--meta_attack', default='Query',
                        help='Query for attack query set only, Query_Support for attack query/support both')
    parser.add_argument('--whether_denoising', default=True,
                        help='whether import denoise block')
    parser.add_argument('--filter_type', default='Gaussian_Filter',
                        help='denoising filter, Median_Filter, Mean_Filter, Gaussian_Filter')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    opt = parser.parse_args()

    (dataset_test, data_loader) = get_dataset(opt)

    config = {
        'epsilon': opt.attack_epsilon,
        'num_steps': opt.attack_steps,
        'step_size': opt.attack_step_size,
        'targeted': opt.attack_targeted,
        'random_init': True
    }

    dloader_test = data_loader(
        dataset=dataset_test,
        nKnovel=opt.way,
        nKbase=0,
        nExemplars=opt.shot,  # num training examples per novel category
        nTestNovel=opt.query * opt.way,  # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=1,
        num_workers=1,
        epoch_size=opt.episode,  # num of batches per epoch
    )

    set_gpu(opt.gpu)
    print('==> Building teacher model..: ', opt.model)
    if opt.model == 'ProtoNet':
        model = ProtoNetEmbedding(activation=opt.activation,
                                  whether_denoising=opt.whether_denoising,
                                  filter_type=opt.filter_type).to(device)
    elif opt.model == 'R2D2':
        model = R2D2Embedding(denoise=opt.denoise, activation=opt.activation,
                              whether_denoising=opt.whether_denoising,
                              filter_type=opt.filter_type).to(device)
    elif opt.model == 'ResNet':
        if opt.dataset == 'miniImageNet' or opt.dataset == 'tieredImageNet':
            model = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=5,
                             whether_denoising=opt.whether_denoising,
                             filter_type=opt.filter_type).to(device)
        else:
            model = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=2,
                             whether_denoising=opt.whether_denoising,
                             filter_type=opt.filter_type).to(device)

    if opt.head == 'ProtoNet':
        cls_head = ClassificationHead(base_learner='ProtoNet').to(device)
    elif opt.head == 'Ridge':
        cls_head = ClassificationHead(base_learner='Ridge').to(device)
    elif opt.head == 'R2D2':
        cls_head = ClassificationHead(base_learner='R2D2').to(device)
    elif opt.head == 'SVM':
        cls_head = ClassificationHead(base_learner='SVM-CS').to(device)
    else:
        print("Cannot recognize the dataset type")
        assert False

    print('==> Loading model..: ')
    for epoch in ['50', '100']:
        for AT in [True, False]:
            checkpoint_path = './experiments/{}/{}/{}/AQ_TRADE'.format(opt.meta_attack, opt.dataset, opt.model)
            saved_models = torch.load(
                checkpoint_path + '/dataset_{}_model_{}_epoch_{}.pth'.format(opt.dataset, opt.model, epoch))
            model.load_state_dict(saved_models['embedding'])
            model.eval()
            cls_head.load_state_dict(saved_models['head'])
            cls_head.eval()
            embedding_net = model

            test_accuracies = []
            for i, batch in enumerate(tqdm(dloader_test()), 1):
                data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]

                n_support = opt.way * opt.shot
                n_query = opt.way * opt.query

                emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
                emb_support = emb_support.reshape(1, n_support, -1)

                data_query_adv = AttackPGD(AT, embedding_net, cls_head, config, data_query, emb_support,
                                           labels_query, labels_support, opt.way, opt.shot, opt.head, 1, n_query)

                emb_query = embedding_net(data_query_adv.reshape([-1] + list(data_query.shape[-3:])))
                emb_query = emb_query.reshape(1, n_query, -1)

                if opt.head == 'SVM':
                    logits = cls_head(emb_query, emb_support, labels_support, opt.way, opt.shot, maxIter=3)
                else:
                    logits = cls_head(emb_query, emb_support, labels_support, opt.way, opt.shot)

                acc = count_accuracy(logits.reshape(-1, opt.way), labels_query.reshape(-1))
                test_accuracies.append(acc.item())

                avg = np.mean(np.array(test_accuracies))
                std = np.std(np.array(test_accuracies))
                ci95 = 1.96 * std / np.sqrt(i + 1)

                if i % 50 == 0:
                    print('Episode [{}/{}]:\t\t\tAccuracy: {:.2f} ± {:.2f} % ({:.2f} %)'
                          .format(i, opt.episode, avg, ci95, acc))

            if not os.path.isdir(checkpoint_path):
                os.makedirs(checkpoint_path + '/', )
            log_file_path = os.path.join(checkpoint_path, "ICML_eval.txt")
            log(log_file_path, 'epoch: {} \t AT={} \t model: {} \t Accuracy: {:.2f} % ({:.2f} %)'.
                format(epoch, str(AT), opt.model, float(np.mean(np.array(test_accuracies))), float(ci95)))
