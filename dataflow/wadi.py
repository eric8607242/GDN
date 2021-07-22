import os
import os.path as osp
import random

import numpy as np
import pandas as pd

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Subset

from .time_dataset import TimeDataset

def get_wadi_dataloader(dataset_path, input_size, batch_size, num_workers, train_portion, slide_win, slide_stride):
    df_train_data = pd.read_csv(osp.join(dataset_path, "train.csv"), sep=",", index_col=0)
    df_test_data = pd.read_csv(osp.join(dataset_path, "test.csv"), sep=",", index_col=0)

    feature_list = get_feature_list(dataset_path)
    fully_connect_struc = get_fully_connect_graph_struc(feature_list)

    edge_indexs = get_edge_indexs(fully_connect_struc, feature_list)

    train_data = construct_data_with_label(df_train_data, feature_list, labels=0)
    test_data = construct_data_with_label(df_test_data, feature_list, labels=df_test_data["attack"])

    train_dataset = TimeDataset(train_data, slide_win, slide_stride)
    test_dataset = TimeDataset(test_data, slide_win)

    train_dataset, val_dataset = get_train_val_dataset(train_dataset, train_portion)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, len(feature_list), edge_indexs

def get_feature_list(dataset_path):
    feature_list_file = open(osp.join(dataset_path, "list.txt"), "r")
    feature_list = []
    
    for f in feature_list_file:
        feature_list.append(f.strip())

    return feature_list


def get_fully_connect_graph_struc(feature_list):
    struc_map = {}
    for f in feature_list:
        if f not in struc_map:
            struc_map[f] = []

        for other_f in feature_list:
            if other_f is not f:
                struc_map[f].append(other_f)

    return struc_map

def get_edge_indexs(struc_map, feature_list):
    index_feature_list = feature_list
    edge_indexs = [[], []]

    for node_name, node_list in struc_map.items():
        source_index = index_feature_list.index(node_name)
        for target in node_list:
            target_indx = index_feature_list.index(target)

            edge_indexs[0].append(source_index)
            edge_indexs[1].append(target_indx)

    edge_indexs = torch.tensor(edge_indexs)
    return edge_indexs

def construct_data_with_label(data, feature_list, labels):
    if "attack" not in data.columns:
        feature_list.append("attack")

        data["attack"] = labels
        data = data[feature_list]
    return data.values


def get_train_val_dataset(dataset, train_portion):
    dataset_len = len(dataset)

    train_dataset_len = int(dataset_len * train_portion)
    val_dataset_len = int(dataset_len * (1 - train_portion))

    val_start_index = random.randrange(train_dataset_len)
    indices = torch.arange(dataset_len)

    train_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_dataset_len:]])
    val_indices = indices[val_start_index:val_start_index+val_dataset_len]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset
