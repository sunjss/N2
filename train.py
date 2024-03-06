from __future__ import division
from __future__ import print_function

from training_setting import args
from utils.utils import N2DataLoader, get_n_params
from utils.loss import bcelogits_loss, weighted_cross_entropy, float_bcelogits_loss
from model.modules import N2Node, N2Graph

import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch
print(f"running on GPU{args.cuda_num}")
print(f"layer: {args.nlayers}; n_pnode: {args.n_pnode}")
print(f"q_dim: {args.q_dim}; n_copies: {args.n_q}; hidden: {args.d_model}")
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

dataset_list = ["ogbn-arxiv", "ogbn-proteins"
                "AmazonComputers", "AmazonPhoto", "CoauthorCS", "CoauthorPhysics",
                'amazon-ratings', 'minesweeper', 'tolokers', 'questions',
                "arxiv-year", "genius"]
def lambda_lr(s):
    s += 1
    if s < args.warmup * args.nbatch:
        return float(s) / float(args.warmup * args.nbatch)
    return max(args.lr_lb, args.lr_step ** (s - args.warmup * args.nbatch))


def model_init(dataset):
    global nfeats
    global nclass
    pre_encoder = None
    pos_encoder = None
    if dataset in dataset_list:
        N2 = N2Node
    else:
        N2 = N2Graph
    model = N2(T=args.nlayers,
                d_in=nfeats,
                d_ein=nedgefeats,
                d_model=args.d_model,
                nclass=nclass, 
                q_dim=args.q_dim,
                n_q=args.n_q,
                n_c=args.n_q if dataset == "ogbg-molpcba" else 1,
                n_pnode=args.n_pnode,
                task_type=task_type,
                dropout=args.dropout,
                self_loop=~args.wo_selfloop,
                pre_encoder=pre_encoder,
                pos_encoder=pos_encoder)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if task_type in ["multi-class", "binary-class"]:
        loss_func = float_bcelogits_loss
    else:
        loss_func = nn.NLLLoss()
    if args.cuda:
        model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model, optimizer, loss_func


class GradCollector(object):
    def __init__(self):
        self.grads = {}

    def __call__(self, name: str):
        def hook(grad):
            self.grads[name] = grad
        return hook


def model_train(model, optimizer, scheduler):
    model.train()
    torch.autograd.set_detect_anomaly(True)
    running_loss = .0
    acc = .0
    div = .0
    out_ls = []
    label_ls = []
    for it, data in enumerate(train_data):
        optimizer.zero_grad()
        grad_collector = GradCollector()
        if isinstance(data, list):
            labels = [d.y for d in data]
            labels = torch.cat(labels).squeeze(-1)
            if labels.ndim == 0:
                    labels = labels.view(1)
            data = Batch.from_data_list(data)
            if args.cuda:
                labels = labels.cuda()
                data = data.cuda()
        else:
            if args.cuda:
                data = data.cuda()
            labels = data.y.squeeze(-1)
        output = model(data)
        if args.dataset in dataset_list:
            train_mask = data.train_mask
            if args.dataset in ['yelp-chi', 'twitch-e', 'ogbn-proteins', 'genius', 'minesweeper', 'tolokers', 'questions']:
                output = output.squeeze(-1)
                loss_train = loss_func(output[train_mask], labels[train_mask])
                acc_train = torch.tensor([0])
                out_ls.append(output[train_mask])
                label_ls.append(labels[train_mask])
            else:
                loss_train = loss_func(output[train_mask], labels[train_mask])
                acc_train = metric(output[train_mask], labels[train_mask])
        elif args.dataset in ["ogbg-molpcba"]:
            is_labeled = labels == labels
            output = output[is_labeled]
            labels = labels[is_labeled]
            out_ls.append(output)
            label_ls.append(labels)
            loss_train = loss_func(output, labels)
            acc_train = torch.tensor([0])
        else:
            loss_train = loss_func(output, labels)
            acc_train = metric(output, labels)
        loss_train.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss_train.data.item()
        acc += acc_train.item()
        div += 1
        tqdm_t.set_description('lr %.4e, %d/%d %.4e' %
                               (scheduler.get_last_lr()[0],
                               it, len(train_data),
                               running_loss / div) +\
                               val_des + sum_des)
        writer.add_scalar('data/train_loss', running_loss / div, epoch * len(train_data) + it)
        writer.add_scalar('data/train_acc', acc / div, epoch * len(train_data) + it)
    if args.dataset in ["ogbg-molpcba"]:
        acc = 0
    elif args.dataset in ['yelp-chi', 'twitch-e', 'ogbn-proteins', 'genius', 'minesweeper', 'tolokers', 'questions']:
        acc = metric(out_ls, label_ls)
    return running_loss / div, acc / div


def model_val(model):
    # Evaluate validation set performance separately,
    # deactivates dropout during validation run.
    model.eval()
    running_loss = .0
    acc = .0
    div = .0
    out_ls = []
    label_ls = []
    with torch.no_grad():
        for it, data in enumerate(val_data):
            if isinstance(data, list):
                labels = [d.y for d in data]
                labels = torch.cat(labels).squeeze(-1)
                if labels.ndim == 0:
                    labels = labels.view(1)
                data = Batch.from_data_list(data)
                if args.cuda:
                    labels = labels.cuda()
                    data = data.cuda()
            else:
                if args.cuda:
                    data = data.cuda()
                labels = data.y.squeeze(-1)
            output = model(data)
            if args.dataset in dataset_list:
                val_mask = data.val_mask
                if args.dataset in ['yelp-chi', 'twitch-e', 'ogbn-proteins', 'genius', 'minesweeper', 'tolokers', 'questions']:
                    output = output.squeeze(-1)
                    loss_val = loss_func(output[val_mask], labels[val_mask])
                    acc_val = torch.tensor([0])
                    out_ls.append(output[val_mask])
                    label_ls.append(labels[val_mask])
                else:
                    loss_val = loss_func(output[val_mask], labels[val_mask])
                    acc_val = metric(output[val_mask], labels[val_mask])
            elif args.dataset in ["ogbg-molpcba"]:
                is_labeled = labels == labels
                output = output[is_labeled]
                labels = labels[is_labeled]
                out_ls.append(output)
                label_ls.append(labels)
                loss_val = loss_func(output, labels)
                acc_val = torch.tensor([0])
            else:
                loss_val = loss_func(output, labels)
                acc_val = metric(output, labels)
            if args.cuda:
                torch.cuda.empty_cache()
            acc += acc_val.item()
            running_loss += loss_val.data.item()
            div += 1
            tqdm_t.set_description('lr %.4e' % (scheduler.get_last_lr()[0]) + \
                                    train_des + ', val %d/%d %f' % 
                                    (it, len(val_data), running_loss / div)+ \
                                    sum_des)
            writer.add_scalar('data/val_loss', running_loss / div, epoch * len(val_data) + it)
            writer.add_scalar('data/val_acc', acc / div, epoch * len(val_data) + it)
    if len(val_data) > 1:
        writer.add_scalar('data/epoch_val_loss', running_loss / div, epoch)
        writer.add_scalar('data/epoch_val_acc', acc / div, epoch)
    if args.dataset in ["ogbg-molpcba"]:
        acc = 0
    elif args.dataset in ['yelp-chi', 'twitch-e', 'ogbn-proteins', 'genius', 'minesweeper', 'tolokers', 'questions']:
        acc = metric(out_ls, label_ls)
    return running_loss / div, acc / div

def model_test(model):
    model.eval()
    running_loss = .0
    acc = .0
    div = .0
    out_ls = []
    label_ls = []
    with tqdm(desc='Test', unit='it', total=len(test_data)) as pbar, torch.no_grad():
        for it, data in enumerate(test_data):
            if isinstance(data, list):
                labels = [d.y for d in data]
                labels = torch.cat(labels).squeeze(-1)
                if labels.ndim == 0:
                    labels = labels.view(1)
                data = Batch.from_data_list(data)
                if args.cuda:
                    labels = labels.cuda()
                    data = data.cuda()
            else:
                if args.cuda:
                    data = data.cuda()
                labels = data.y.squeeze(-1)
            output = model(data)
            if args.dataset in dataset_list:
                test_mask = data.test_mask
                if args.dataset in ['yelp-chi', 'twitch-e', 'ogbn-proteins', 'genius', 'minesweeper', 'tolokers', 'questions']:
                    output = output.squeeze(-1)
                    loss_test = loss_func(output[test_mask], labels[test_mask])
                    acc_test = torch.tensor([0])
                    out_ls.append(output[test_mask])
                    label_ls.append(labels[test_mask])
                else:
                    loss_test = loss_func(output[test_mask], labels[test_mask])
                    acc_test = metric(output[test_mask], labels[test_mask])
            elif args.dataset in ["ogbg-molpcba"]:
                is_labeled = labels == labels
                output = output[is_labeled]
                labels = labels[is_labeled]
                out_ls.append(output)
                label_ls.append(labels)
                loss_test = loss_func(output, labels)
                acc_test = torch.tensor([0])
            else:
                loss_test = loss_func(output, labels)
                acc_test = metric(output, labels)
            if args.cuda:
                torch.cuda.empty_cache()
            acc += acc_test.item()
            running_loss += loss_test.data.item()
            div += 1
            pbar.update()
    if args.dataset in ["ogbg-molpcba"]:
        acc = len(test_data) * metric(out_ls, label_ls)
    elif args.dataset in ['yelp-chi', 'twitch-e', 'ogbn-proteins', 'genius', 'minesweeper', 'tolokers', 'questions']:
        acc = metric(out_ls, label_ls)
    if type(acc) is dict:
        print("Test set results:",
                "loss= {:.4f}".format(running_loss / div))
        acc_out = ""
        for k, v in acc.items():
            acc_out += f"{k}={v / div}, "
        acc = acc_out[:-2]
        print(acc)
    else:
        print("Test set results:",
                "loss= {:.4f}".format(running_loss / div),
                "metric= {:.4f}".format(acc / div))
        acc = round(acc / div, 5)

    return round(running_loss / div, 5), acc


if __name__ == '__main__':
    # Load data
    data_loader = N2DataLoader()
    print("Loading " + args.dataset.title() + "...")
    flag = (args.dataset in ["NCI1", "IMDB-BINARY", "IMDB-MULTI", "PROTEINS", "COLLAB"])
    graph_iter_range = 10 if flag else 1
    for k in range(graph_iter_range):
        args.fold_idx = k if graph_iter_range > 1 else args.fold_idx
        data_loader.load_data(dataset=args.dataset, spilit_type="public", 
                            nbatch=args.nbatch, fold_idx=args.fold_idx)
        nclass, nfeats, nedgefeats = data_loader.nclass, data_loader.nfeats, data_loader.nedgefeats
        train_data, val_data, test_data = data_loader.train_data, data_loader.val_data, data_loader.test_data
        metric = data_loader.metric
        task_type = data_loader.task_type

        save_path = 'export/' + args.testmode + args.dataset + '/'
        model_config =  '%.1f_%d_%d_%d_%d_%d/' % (args.dropout, args.d_model, args.n_pnode, args.nlayers, args.q_dim, args.n_q)
        if args.dataset in ["NCI1", "IMDB-BINARY", "IMDB-MULTI", "PROTEINS", "COLLAB"]:
            model_config = model_config + '%d/' % (args.fold_idx)
            print("Running on split " + str(args.fold_idx))
        save_path = save_path + model_config
        is_exists = os.path.exists(save_path)
        if not is_exists:
            save_path = save_path + "1/"
            os.makedirs(save_path)
        else:
            files = os.listdir(save_path)
            if args.resume_last or args.resume_best:
                file_idx = str(len(files))
                save_path = save_path + file_idx + "/"
            else:
                file_idx = str(len(files) + 1)
                save_path = save_path + file_idx + "/"
                os.makedirs(save_path)
        model, optimizer, loss_func = model_init(args.dataset)
        print('number of parameters:', get_n_params(model))
        name = "N2"
        logfile = "node as neuron"
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda_lr)
        continue_flag = True
        bad_counter = 0
        start_epoch = 0
        best = args.epochs + 1
        writer = SummaryWriter(log_dir=os.path.join(save_path, name))
        if args.resume_last or args.resume_best:
            if args.resume_last:
                fname = save_path + logfile + '_last.pth'
            else:
                fname = save_path + logfile + '_best.pth'

            if os.path.exists(fname):
                data = torch.load(fname)
                torch.set_rng_state(data['torch_rng_state'])
                torch.cuda.set_rng_state(data['cuda_rng_state'])
                np.random.set_state(data['numpy_rng_state'])
                random.setstate(data['random_rng_state'])
                model.load_state_dict(data['state_dict'], strict=False)
                optimizer.load_state_dict(data['cocn_optimizer'])
                scheduler.load_state_dict(data['cocn_scheduler'])
                start_epoch = data['epoch'] + 1
                bad_counter = data['patience']
                best = data['best_val_loss']
                print('Resuming from epoch %d, best validation loss %f' % (
                    data['epoch'], data['best_val_loss']))
        val_des = ', val NaN'
        sum_des = ', bad 0, best NaN'
        l_bad_cnt = -1
        print("Start training...")
        with tqdm(unit='it', total=args.epochs) as tqdm_t:
            for epoch in range(start_epoch, start_epoch + args.epochs):
                if not continue_flag:
                    break
                loss_train_value, acc_train_value = model_train(model, optimizer, scheduler)
                train_des = ', %d/%d %.4e' % (len(train_data), len(train_data), loss_train_value)
                torch.save({
                    'torch_rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state(),
                    'numpy_rng_state': np.random.get_state(),
                    'random_rng_state': random.getstate(),
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'cocn_optimizer': optimizer.state_dict(),
                    'cocn_scheduler': scheduler.state_dict(),
                    'patience': bad_counter,
                    'best_val_loss': best,
                }, save_path + logfile + "_last.pth")
                if not args.fastmode:
                    loss_val_value, acc_val_value = model_val(model)
                    if loss_val_value < best:
                        best = loss_val_value
                        torch.save({
                            'torch_rng_state': torch.get_rng_state(),
                            'cuda_rng_state': torch.cuda.get_rng_state(),
                            'numpy_rng_state': np.random.get_state(),
                            'random_rng_state': random.getstate(),
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'cocn_optimizer': optimizer.state_dict(),
                            'cocn_scheduler': scheduler.state_dict(),
                            'patience': bad_counter,
                            'best_val_loss': best,
                        }, save_path + logfile + "_best.pth")
                        if bad_counter > l_bad_cnt:
                            l_bad_cnt = bad_counter
                        bad_counter = 0
                    else:
                        bad_counter += 1
                    val_des = ', val %d/%d %f' % (len(val_data), len(val_data), loss_val_value)
                    sum_des = ', bad %i, best %.4f' % (bad_counter, best)
                    tqdm_t.set_description('lr %.4e' % (scheduler.get_last_lr()[0]) + \
                                            train_des + val_des + sum_des)
                    continue_flag = (bad_counter <= args.patience) or (l_bad_cnt == -1)
                tqdm_t.update(1)
        if not args.fastmode:
            data = torch.load(save_path + logfile + "_best.pth")
            model.load_state_dict(data['state_dict'])
            print("Testing model..." + str(l_bad_cnt))
            test_loss, test_acc = model_test(model)
            

        print('Exporting data......')
        write_text = "-------training arg-------" + '\n'
        for k, v in vars(args).items():
            if k == "nbatch":
                write_text += "-------dataset arg-------\n"
            elif k == "nlayers":
                write_text += "-------model arg-------\n"
            write_text += f'{k} = {v}\n'
        write_text = write_text + "-------results-------\n"
        write_text = write_text + 'loss = ' + str(test_loss) + '\n' + 'acc = ' + str(test_acc) + '\n'
        write_text = write_text + 'best val loss = ' + str(best) + '\n'
        fname = '/' + args.dataset + '_result.txt'
        with open(save_path + fname, "w") as f:
            f.write(write_text)
        f.close()
    print("Done!")