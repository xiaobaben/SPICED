import copy
import torch
import argparse
import numpy as np
import os
from utils.util import fix_randomness, compute
from torch.utils.data import DataLoader
from dataloader.data_loader import Builder
from builder.synaptic_builder import SynapticNetwork
from trainer.pretrainer import pretraining
from trainer.trainer import trainer


def get_save_path(param):
    return f'source_percent{param.source_percent}' \
           f'_dataset_{param.dataset}' \
           f'_seed_{param.seed}' \
           f'_stability_{param.stability}' \
           f'_alpha_sim_{param.alpha_sim}' \
           f'_alpha_str_{param.alpha_str}' \
           f'_k_factor_{param.k_factor}' \
           f'_decay_factor_{param.decay_factor}' \
           f'_consolidation_factor_{param.consolidation_factor}' \
           f'threshold{param.similarity_threshold}' \
           f'prototype{param.prototype}' \
           f'weight_new_cls{param.weight_new_cls}' \
           f'replay_strategy{param.replay_strategy}'


def get_prototype_path(param):
    if param.dataset == "ISRUC":
        return f"/data/xbb/Synaptic Homeostasis/dataset/ISRUC/{param.prototype}"
    if param.dataset == "FACED":
        return f"/data/xbb/Synaptic Homeostasis/dataset/FACED/{param.prototype}"
    if param.dataset == "BCI2000":
        return f"/data/xbb/Synaptic Homeostasis/dataset/BCI2000/{param.prototype}"


def get_filepath(dataset):
    path = None
    # Here, return the actual data storage address.
    if dataset == "ISRUC":
        path = f"/data/datasets2/ISRUC_Group1/EEGEOG"
    elif dataset == "FACED":
        path = f"/data/datasets2/Face"
    elif dataset == 'BCI2000':
        path = f"/data/datasets2/BCI2000-4"
    return path


def get_path_loader(params):
    path = None
    if params.dataset == 'ISRUC':
        path = [i for i in range(1, 101) if i not in [8, 40]]
    elif params.dataset == "FACED":
        path = [i for i in range(123)]
    elif params.dataset == 'BCI2000':
        path = [i for i in range(109) if i not in [38, 88, 89, 92, 100, 104]]

    path_name = {int(j): [[], []] for j in path}

    for t_idx in path:
        num = 0
        file_path = params.file_path + f"/{t_idx}/data"
        label_path = params.file_path + f"/{t_idx}/label"
        while os.path.exists(file_path + f"/{num}.npy"):
            path_name[t_idx][0].append(file_path + f"/{num}.npy")
            path_name[t_idx][1].append(label_path + f"/{num}.npy")
            num += 1

    return path, path_name


def get_idx(params, path):
    fix_randomness(params.seed)
    idx = path
    path_len = len(idx)

    train_val_idx = list(np.random.choice(idx, int(path_len*params.source_percent), replace=False))
    incremental_idx = list(set(idx)-set(train_val_idx))
    train_idx = list(np.random.choice(train_val_idx, int(len(train_val_idx)*0.8), replace=False))
    params.train_num = len(train_idx)
    val_idx = [i for i in train_val_idx if i not in train_idx]

    print(" Train Idx  ", len(train_idx), sorted(train_idx), "\n",
          "Val Idx  ", len(val_idx), sorted(val_idx), "\n",
          "Incremental Idx", len(incremental_idx), sorted(incremental_idx), "\n",)

    return train_idx, val_idx, incremental_idx, train_val_idx


def get_loader(params, path, path_name):
    train_path = [[], []]
    val_path = [[], []]
    train_idx, val_idx, incremental_idx, train_val_idx = get_idx(params, path)

    for t_idx in train_idx:
        train_path[0].extend(path_name[t_idx][0])
        train_path[1].extend(path_name[t_idx][1])

    for v_idx in val_idx:
        val_path[0].extend(path_name[v_idx][0])
        val_path[1].extend(path_name[v_idx][1])

    train_builder = Builder(train_path, params).Dataset
    val_builder = Builder(val_path, params).Dataset

    return train_builder, val_builder, incremental_idx, train_val_idx


def main():
    parser = argparse.ArgumentParser(description='Synaptic-Inspired Continual Brain Decoding')
    parser.add_argument('--pretrain_epoch', type=int, default=100, help='pretrain epoch')
    parser.add_argument('--incremental_epoch', type=int, default=20,  help='incremental epoch')
    parser.add_argument('--dataset', type=str, default='BCI2000', help='dataset choosing [ISRUC, FACED, BCI2000]')
    parser.add_argument('--gpu', type=int, default=2, help='cuda number(default:0)')
    parser.add_argument('--seed', type=int, default=4321, help='random seed')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--cl_lr', type=float, default=1e-6, help='continual learning rate')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1')
    parser.add_argument('--beta2', type=float, default=0.99, help='beta2')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--num_worker', type=int, default=4, help='num worker')
    parser.add_argument('--is_pretrain', type=bool, default=False, help='pretraining')
    parser.add_argument('--source_percent', type=float, default=0.3, help='source individual percent')
    parser.add_argument('--times', type=list, default=[], help='time cost')
    """Synaptic Network hyper-parameters"""
    parser.add_argument('--stability', type=float, default=None, help='stability')
    parser.add_argument('--decay_factor', type=float, default=30, help='renormalization_synapses, Ebbinghaus S')
    parser.add_argument('--consolidation_factor', type=float, default=1.3, help='consolidate_memory')
    parser.add_argument('--alpha_sim', type=float, default=0.6, help='importance coefficient of similarity')
    parser.add_argument('--k_factor', type=int, default=15, help='find importance knn')
    parser.add_argument('--similarity_threshold', type=float, default=0.1, help='connected similarity threshold')
    parser.add_argument('--saving_confidence', type=float, default=0.8, help='saving pseudo labeled threshold')
    parser.add_argument('--training_confidence', type=float, default=0.9, help='training preserved threshold')
    parser.add_argument('--weight_new_cls', type=float, default=0.7, help='loss weight of replay & incremental data')
    parser.add_argument('--prototype', type=str, default='prototype3', help='prototype')
    parser.add_argument('--replay_strategy', type=str, default='cls', help='replay data strategy')

    params = parser.parse_args()
    parser.add_argument('--alpha_str', type=float, default=1-params.alpha_sim, help='importance coefficient of average strength')
    params = parser.parse_args()
    parser.add_argument('--file_path', type=str, default=get_filepath(params.dataset), help='data file path')
    parser.add_argument('--save_info', type=str, default=get_save_path(params), help='saving model path')
    parser.add_argument('--prototype_path', type=str, default=get_prototype_path(params), help="prototype path")
    params = parser.parse_args()
    print(params)
    fix_randomness(params.seed)
    path, path_name = get_path_loader(params)

    train_dataset, val_dataset, new_task_idx, source_idx = get_loader(params, path, path_name)

    train_loader = DataLoader(dataset=train_dataset, batch_size=params.batch,
                              shuffle=False, num_workers=params.num_worker)
    val_loader = DataLoader(dataset=val_dataset, batch_size=params.batch,
                            shuffle=True, num_workers=params.num_worker)

    if params.is_pretrain:
        pretraining(train_loader, val_loader, params)

    """Synaptic Network Initialization"""
    synaptic_net = SynapticNetwork(params)
    initial_info = []
    for source_id in source_idx:
        sub_name = f'Sub_{source_id}'
        sub_prototype = np.load(params.prototype_path + f"/{source_id}.npy")
        sub_data_path = path_name[source_id][0]
        sub_label_path = path_name[source_id][1]
        initial_info.append([sub_name, sub_prototype, sub_data_path, sub_label_path])
        synaptic_net.net_initialization(initial_info, f"model_parameter/{params.dataset}/Pretrain/{params.source_percent}/",
                                        params.seed, similarity_threshold=params.similarity_threshold)

    source_net = copy.deepcopy(synaptic_net)

    result, synaptic_net = trainer(new_task_idx, params, synaptic_net, source_net)
    print(result)
    compute(result)


if __name__ == '__main__':
    """
    Step1: Pretraining the source model; Any decoding model can be chosen, e.g., EEGNet, EEGConformer
    Step2: To facilitate computing, we need to compute all individuals' initial feature first
    Step3: Begin Continual EEG Decoding
    """
    torch.multiprocessing.set_start_method('spawn')
    main()




