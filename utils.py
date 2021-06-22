import os
import time
import pprint
import torch
import torch.nn.functional as F
import random
import pandas as pd
import matplotlib.pyplot as plt


def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def check_dir(path):
    '''
    Create directory if it does not exist.
        path:           Path of directory.
    '''
    if not os.path.exists(path):
        os.mkdir(path)


def count_accuracy(logits, label):
    pred = torch.argmax(logits, dim=1).view(-1)
    label = label.view(-1)
    accuracy = 100 * pred.eq(label).float().mean()
    return accuracy


class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / float(p)
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


def log(log_file_path, string):
    '''
    Write one line of log into screen and file.
        log_file_path: Path of log file.
        string:        String to write in log file.
    '''
    with open(log_file_path, 'a+') as f:
        f.write(string + '\n')
        f.flush()
    print(string)


def AttackPGD(attack, embedding_net, cls_head, config, data_query, emb_support, labels_query, labels_support, way, shot,
              head, episodes_per_batch, n_query, maxIter=3):
    if not attack:
        return data_query
    if config['targeted']:
        new_labels_query = torch.zeros_like(labels_query)
        for i in range(int(labels_query.size()[0])):
            for j in range(int(labels_query.size()[1])):
                while True:
                    new_labels_query[i, j] = random.randint(0, way - 1)
                    if new_labels_query[i, j] != labels_query[i, j]:
                        break
    else:
        new_labels_query = labels_query
    new_labels_query = new_labels_query.view(new_labels_query.size()[0] * new_labels_query.size()[1])
    x = data_query.detach()
    if config['random_init']:
        x = x + torch.zeros_like(x).uniform_(-config['epsilon'], config['epsilon'])
    for i in range(config['num_steps']):
        x.requires_grad_()
        with torch.enable_grad():
            emb_query_adv = embedding_net(x.reshape([-1] + list(x.shape[-3:]))).reshape(episodes_per_batch, n_query, -1)

            if head == 'SVM':
                logits = cls_head(emb_query_adv, emb_support, labels_support, way, shot, maxIter=maxIter)
            else:
                logits = cls_head(emb_query_adv, emb_support, labels_support, way, shot)

            logits = logits.view(logits.size()[0] * logits.size()[1], logits.size()[2])
            loss = F.cross_entropy(logits, new_labels_query, size_average=False)
        grad = torch.autograd.grad(loss, [x])[0]
        if config['targeted']:
            x = x.detach() - config['step_size'] * torch.sign(grad.detach())
        else:
            x = x.detach() + config['step_size'] * torch.sign(grad.detach())
        x = torch.min(torch.max(x, data_query - config['epsilon']), data_query + config['epsilon'])
        x = torch.clamp(x, 0.0, 1.0)
    return x


def AttackPGDFeatureExtractor(attack, embedding_net, cls_head, config, data_query, labels_query, ways=64, maxIter=3):
    if not attack:
        return data_query
    if config['targeted']:
        new_labels_query = torch.zeros_like(labels_query)
        for i in range(int(labels_query.size()[0])):
            while True:
                new_labels_query[i] = random.randint(0, ways - 1)
                if new_labels_query[i] != labels_query[i]:
                    break
    else:
        new_labels_query = labels_query

    x = data_query.detach()
    if config['random_init']:
        x = x + torch.zeros_like(x).uniform_(-config['epsilon'], config['epsilon'])
    for i in range(config['num_steps']):
        x.requires_grad_()
        with torch.enable_grad():
            emb_query_adv = embedding_net(x)
            logits = cls_head(emb_query_adv)
            loss = F.cross_entropy(logits, new_labels_query)
        grad = torch.autograd.grad(loss, [x])[0]
        if config['targeted']:
            x = x.detach() - config['step_size'] * torch.sign(grad.detach())
        else:
            x = x.detach() + config['step_size'] * torch.sign(grad.detach())
        x = torch.min(torch.max(x, data_query - config['epsilon']), data_query + config['epsilon'])
        x = torch.clamp(x, 0.0, 1.0)
    return x


"""
data_support_teacher_adv = AttackPGD(opt.attack_embedding, teacher_model, cls_teacher_head, config, data_support,
                               emb_support_teacher, labels_query, labels_support, opt.train_way,
                               opt.train_shot, opt.teacher_head, opt.episodes_per_batch, train_n_query)

emb_support_teacher_adv = teacher_model(data_support_teacher_adv.reshape([-1] + list(data_support.shape[-3:])))
emb_support_teacher_adv = emb_support_teacher_adv.reshape(opt.episodes_per_batch, train_n_support, -1)
"""

"""
logit_query_teacher = cls_teacher_head(emb_query_teacher, emb_support_teacher_adv, labels_support,
                                       opt.train_way, opt.train_shot)
"""


def draw_figure(path):
    fig, axs = plt.subplots(2, 2)
    df_avg_loss = pd.read_csv(path + 'avg_loss_log.csv', header=None, names=['epoch', 'loss'])
    df_loss = pd.read_csv(path + 'loss_log.csv', header=None, names=['iter', 'loss'])
    df_avg_acc = pd.read_csv(path + 'avg_acc_log.csv', header=None, names=['epoch', 'acc'])
    df_acc = pd.read_csv(path + 'acc_log.csv', header=None, names=['iter', 'acc'])
    axs[0, 0].plot(df_avg_loss['epoch'], df_avg_loss['loss'])
    axs[0, 0].set_title('avg_loss_figure')
    axs[0, 0].set(xlabel='epoch', ylabel='loss')

    axs[0, 1].plot(df_loss['iter'], df_loss['loss'])
    axs[0, 1].set_title('overall_loss_figure')
    axs[0, 1].set(xlabel='iteration')

    axs[1, 0].plot(df_avg_acc['epoch'], df_avg_acc['acc'])
    axs[1, 0].set_title('avg_acc_figure')
    axs[1, 0].set(xlabel='epoch', ylabel='acc')

    axs[1, 1].plot(df_acc['iter'], df_acc['acc'])
    axs[1, 1].set_title('overall_acc_figure')
    axs[1, 1].set(xlabel='iteration')

    fig.tight_layout(pad=2.0)
    fig.savefig(path+"log_figure.png")

def draw_compared_figure(path1, path2, files):
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(5, 5)
    df_1 = pd.read_csv(path1 + files, header=None, names=['x', 'y'])
    df_2 = pd.read_csv(path2 + files, header=None, names=['x', 'y'])


    axs.plot(df_1['x'], df_1['y'])
    axs.plot(df_2['x'], df_2['y'], 'r')
    # axs[0, 0].set_title('avg_loss_figure')
    ylabel = 'loss'
    xlabel = 'iteration'
    if 'acc' in files:
        ylabel = 'acc'
    if 'avg' in files:
        xlabel = 'epoch'
    axs.set(xlabel=xlabel, ylabel=ylabel)
    axs.legend(['Adam', 'SGD'], loc='upper left')

    path = './compared_figures_forgetness/adam_compared_with_SGD_AT_Q_S'
    if not os.path.isdir(path):
        os.makedirs(path + '/')
    # fig.tight_layout(pad=2.0)
    fig.savefig(path+'/{}.png'.format(files[:-4]))


if __name__ == '__main__':
    # path = './checkpoint/Query/adversarial/miniImageNet/ProtoNet/AT_per_10_epoch/'
    # draw_figure(path)
    path1 = './checkpoint/Query_Support/adversarial/miniImageNet/ProtoNet/new_adam/'
    path2 = './checkpoint/Query_Support/adversarial/miniImageNet/ProtoNet/'
    for files in ['acc_log.csv', 'loss_log.csv', 'avg_acc_log.csv', 'avg_loss_log.csv']:
        draw_compared_figure(path1, path2, files)
