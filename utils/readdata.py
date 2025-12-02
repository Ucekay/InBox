import numpy as np
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp

import random
from time import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import os
import torch

n_users = 0
n_items = 0
n_entities = 0
n_relations = 0
n_nodes = 0
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)


def read_cf(file_name):
    '''
    将每个行为记录单独记录下来,(user_id, item_id)
    '''
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]

        u_id, pos_ids = inters[0], inters[1:]
        # 此处原本是将所有的商品只展示一次，但是重复出现的ui pair应该获得更多的训练
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])

    return np.array(inter_mat)


def remap_item(train_data, test_data):
    '''
    user_set以user ID为key，item ID list为value
    '''

    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))
    # test_user_set要去重，后续计算指标的时候用
    for u_id, item_list in test_user_set.items():
        test_user_set[u_id] = list(set(test_user_set[u_id]))
        


def read_triplets(file_name):
    '''
    读取KG的数据，一triplets的形式记录下来
    返回的就是每个三元组的数据，就是final.txt中记录的数据。
    '''
    global n_entities, n_relations, n_nodes

    can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
    can_triplets_np = np.unique(can_triplets_np, axis=0)
    triplets = can_triplets_np.copy()
    
    # 这里加1是因为id是从0开始的，不是padding id
    n_ori_relations = max(triplets[:, 1]) + 1
    n_relations = n_ori_relations * 2

    # 将所有的triplet分成两类：1、IRT(item, relation, tag)，2、TRT(tag, relation, tag)
    triplets_IRT = []
    triplets_TRT = []
    triplets_IRI = []
    for triplet in triplets:
        # TRT和IRI添加了逆三元组，所以预训练只需要一个数据集
        if (triplet[0] >= n_items) and (triplet[2] >= n_items):
            h, r, t = triplet
            triplets_TRT.append(list(triplet))
            triplets_TRT.append([t, r+n_ori_relations, h])
        elif (triplet[0] < n_items) and (triplet[2] < n_items):
            h, r, t = triplet
            triplets_IRI.append(list(triplet))
            triplets_IRI.append([t, r+n_ori_relations, h])
        else:
            # 所有的IRT数据都按照item relation tag的顺序保存
            # IRT 没有添加逆三元组，分别替换item和tag
            h, r, t = triplet
            if (h >= n_items) and (t < n_items):
                # (tag, relation, item) -> (item, relation+n_ori, tag)
                triplets_IRT.append([t, r+n_ori_relations, h])
            elif (h < n_items) and (t >= n_items):
                # (item, relation, tag) -> そのまま
                triplets_IRT.append(list(triplet))
            # else: h < n_items and t < n_items はIRIに分類されるべき（ここには来ない）
            # または h >= n_items and t >= n_items はTRTに分類されるべき（ここには来ない）

    return triplets_IRT, triplets_TRT, triplets_IRI


def map_tag_to_item(triplets, path):
    data = pd.read_csv(path, header=0, sep=' ')
    items = set(data['remap_id'])
    items = set(map(int, items))
    item_tag = defaultdict(list)
    for triplet in triplets:
        if triplet[0] in items:
            item_tag[triplet[0]].append([triplet[1],triplet[2]])
        if triplet[2] in items:
            assert False, "item tag false."

    return item_tag

def get_num(path, name):
    # return len(pd.read_csv(os.path.join(path, name), sep = ' ', error_bad_lines=False))
    return len(open(os.path.join(path, name)).readlines())-1

def build_mat(test_data, test_inter_mat):
    for u_id, i_id in test_data:
        test_inter_mat[int(u_id)][int(i_id)] = 1
    return test_inter_mat

def load_data(model_args):
    global args
    args = model_args
    directory = './data/' + args.dataset + '/'

    global n_users, n_items, n_entities
    n_users = get_num(directory, 'user_list.txt')
    n_items = get_num(directory, 'item_list.txt')
    n_entities = get_num(directory, 'entity_list.txt')

    print('Reading train and test user-item set ...')
    train_cf = read_cf(directory + 'train.txt')
    test_cf = read_cf(directory + 'test.txt')
    remap_item(train_cf, test_cf)

    print('Reading kg data ...')
    triplets_IRT, triplets_TRT, triplets_IRI = read_triplets(directory + 'kg_final.txt')
    print('Mapping tags to items ...')
    item_tag = map_tag_to_item(triplets_IRT, directory + 'item_list.txt')
    # print('building the graph ...')
    # graph, relation_dict = build_graph(train_cf, triplets)

    # print('building the adj mat ...')
    # adj_mat_list, norm_mat_list, mean_mat_list = build_sparse_relational_graph(relation_dict)

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'n_entities': int(n_entities),
        'n_tags': int(n_entities) - int(n_items),
        'n_relations': int(n_relations)
    }


    test_inter_mat = torch.zeros(n_users, n_items)
    test_inter_mat = build_mat(test_cf, test_inter_mat)
    test_inter_mat = test_inter_mat.long()
    # train_cf, test_cf: ui对的array
    # train_user_set, test_user_set: dict, key为user id, value为该user交互过的item id list
    # item_tag: dict, key为item id, value为该item具有的[relation tag] 组成的 list
    # triplets_IRT, triplets_TRT, triplets_IRI: list, 每个元素都为triplet的list
    return train_cf, test_cf, train_user_set, test_user_set, test_inter_mat, item_tag, triplets_IRT, triplets_TRT, triplets_IRI, n_params