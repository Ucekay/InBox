#!/usr/bin/python3


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import random
from collections import defaultdict
import math

from torch.utils.data import Dataset


class PreTrainDataset_IRT(Dataset):
    def __init__(self, triplets, triplets_train, negative_sample_size, n_params, mode):
        self.triplets = triplets
        self.data = triplets_train
        self.len = len(self.data)
        self.negative_sample_size = negative_sample_size
        self.n_items = n_params['n_items']
        self.n_entities = n_params['n_entities']
        self.n_tags = n_params['n_tags']
        self.n_relation = n_params['n_relations']
        self.mode = mode

        self.count = self.count_frequency(self.triplets, self.n_relation)
        self.true_item, self.true_tag = self.get_true_item_and_tag(self.triplets)
        # オンザフライサンプリングに変更: get_neg()の事前計算を削除

        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        # コピーを作成して元のデータを変更しないようにする
        positive_sample = list(self.data[idx])
        item, relation, tag = positive_sample
        
        # 境界チェック: relation が有効範囲内であることを確認
        if relation < 0 or relation >= self.n_relation:
            raise ValueError(f"Invalid relation in IRT: relation={relation}, n_relation={self.n_relation}, idx={idx}")
        
        # KeyError防止: キーが存在しない場合はデフォルト値4を使用
        inverse_relation = (relation + self.n_relation // 2) % self.n_relation
        count_head = self.count.get((item, relation), 4)
        count_tail = self.count.get((tag, inverse_relation), 4)
        subsampling_weight = count_head + count_tail
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        
        if self.mode == 'IRT-item':
            # オンザフライでネガティブサンプルを生成（メモリ効率改善）
            negative_sample_list = []
            negative_sample_size = 0
            true_items = self.true_item.get((relation, tag), np.array([], dtype=np.int64))
            while negative_sample_size < self.negative_sample_size:
                neg_samples = np.random.randint(0, self.n_items, size=self.negative_sample_size*2)
                mask = np.in1d(
                    neg_samples, 
                    true_items, 
                    assume_unique=True,
                    invert=True
                )
                neg_samples = neg_samples[mask]
                negative_sample_list.append(neg_samples)
                negative_sample_size += neg_samples.size
            negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
            
        elif self.mode == 'IRT-tag':
            negative_sample_list = []
            negative_sample_size = 0
            # KeyError防止: true_tagにキーが存在しない場合は空配列を使用
            true_tags = self.true_tag.get((item, relation), np.array([], dtype=np.int64))
            while negative_sample_size < self.negative_sample_size:
                negative_sample = np.random.randint(self.n_items, self.n_entities, size=self.negative_sample_size*2)
                mask = np.in1d(
                    negative_sample, 
                    true_tags, 
                    assume_unique=True,
                    invert=True
                )
                negative_sample = negative_sample[mask]
                negative_sample_list.append(negative_sample)
                negative_sample_size += negative_sample.size
            negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        
        n_tags = self.n_entities - self.n_items
        
        if self.mode == 'IRT-tag':
            negative_sample = negative_sample - self.n_items
            # IRT-tagモード: negative_sampleの境界チェック
            if np.any(negative_sample < 0) or np.any(negative_sample >= n_tags):
                bad_vals = negative_sample[(negative_sample < 0) | (negative_sample >= n_tags)]
                raise ValueError(f"Invalid negative_sample in IRT-tag: values={bad_vals[:5]}, n_tags={n_tags}")
        elif self.mode == 'IRT-item':
            # IRT-itemモード: negative_sampleの境界チェック
            if np.any(negative_sample < 0) or np.any(negative_sample >= self.n_items):
                bad_vals = negative_sample[(negative_sample < 0) | (negative_sample >= self.n_items)]
                raise ValueError(f"Invalid negative_sample in IRT-item: values={bad_vals[:5]}, n_items={self.n_items}")
        
        # 境界チェック: tagがn_items以上であることを確認（IRTトリプルの条件）
        tag_idx = positive_sample[2] - self.n_items
        if tag_idx < 0 or tag_idx >= n_tags:
            raise ValueError(f"Invalid tag index in IRT: original_tag={positive_sample[2]}, "
                           f"tag_idx={tag_idx}, n_items={self.n_items}, n_tags={n_tags}, mode={self.mode}")
        
        negative_sample = torch.LongTensor(negative_sample)
        positive_sample[2] = tag_idx
        positive_sample = torch.LongTensor(positive_sample)
        
        return positive_sample, negative_sample, subsampling_weight, self.mode

    @staticmethod
    def count_frequency(triplets, n_relation, start=4):
        count = {}
        for head, relation, tail in triplets:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, (relation+n_relation//2)%(n_relation)) not in count:
                count[(tail, (relation+n_relation//2)%(n_relation))] = start
            else:
                count[(tail, (relation+n_relation//2)%(n_relation))] += 1
        return count

    @staticmethod
    def get_true_item_and_tag(triples):
        true_item = {}
        true_tag = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tag:
                true_tag[(head, relation)] = []
            true_tag[(head, relation)].append(tail)
            if (relation, tail) not in true_item:
                true_item[(relation, tail)] = []
            true_item[(relation, tail)].append(head)

        for relation, tail in true_item:
            true_item[(relation, tail)] = np.array(list(set(true_item[(relation, tail)])))
        for head, relation in true_tag:
            true_tag[(head, relation)] = np.array(list(set(true_tag[(head, relation)])))                 

        return true_item, true_tag

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode

    # get_neg()メソッドは削除: オンザフライサンプリングに移行
    # 旧実装は大量のメモリを消費していた（全(relation, tag)ペア × 全アイテム数）


class PreTrainDataset_TRT_IRI(Dataset):
    def __init__(self, triplets, negative_sample_size, n_params, mode):
        self.triplets = triplets
        self.len = len(self.triplets)
        self.negative_sample_size = negative_sample_size
        self.n_items = n_params['n_items']
        self.n_entities = n_params['n_entities']
        self.n_tags = n_params['n_tags']
        self.n_relation = n_params['n_relations']
        self.mode = mode

        self.count = self.count_frequency(self.triplets)
        self.true_head, self.true_tail = self.get_true_item_and_tag(self.triplets)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        # コピーを作成して元のデータを変更しないようにする
        positive_sample = list(self.triplets[idx])
        head, relation, tail = positive_sample
        
        # KeyError防止: キーが存在しない場合はデフォルト値4を使用
        inverse_relation = (relation + self.n_relation // 2) % self.n_relation
        count_head = self.count.get((head, relation), 4)
        count_tail = self.count.get((tail, inverse_relation), 4)
        subsampling_weight = count_head + count_tail

        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        negative_sample_list = []
        negative_sample_size = 0

        # KeyError防止: true_tailにキーが存在しない場合は空配列を使用
        true_tails = self.true_tail.get((head, relation), np.array([], dtype=np.int64))
        
        while negative_sample_size < self.negative_sample_size:
            if self.mode == 'TRT':
                negative_sample = np.random.randint(self.n_items, self.n_entities, size=self.negative_sample_size*2)
                mask = np.in1d(
                    negative_sample, 
                    true_tails,
                    assume_unique=True,
                    invert=True
                )
            elif self.mode == 'IRI':
                negative_sample = np.random.randint(0, self.n_items, size=self.negative_sample_size*2)
                mask = np.in1d(
                    negative_sample, 
                    true_tails, 
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('PreTraining batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        
        # 境界チェック
        if self.mode == 'TRT':
            # TRTモード: head, tailはどちらもタグ（>= n_items）
            if positive_sample[0] < self.n_items or positive_sample[2] < self.n_items:
                raise ValueError(f"Invalid TRT triplet: head={positive_sample[0]}, tail={positive_sample[2]}, n_items={self.n_items}")
            positive_sample[2] = positive_sample[2] - self.n_items
            positive_sample[0] = positive_sample[0] - self.n_items
            negative_sample = negative_sample - self.n_items
            
            # 変換後の値が範囲内であることを確認
            n_tags = self.n_entities - self.n_items
            if positive_sample[0] < 0 or positive_sample[0] >= n_tags or positive_sample[2] < 0 or positive_sample[2] >= n_tags:
                raise ValueError(f"Invalid TRT tag indices after conversion: head={positive_sample[0]}, tail={positive_sample[2]}, n_tags={n_tags}")
        elif self.mode == 'IRI':
            # IRIモード: head, tailはどちらもアイテム（< n_items）
            if positive_sample[0] >= self.n_items or positive_sample[2] >= self.n_items:
                raise ValueError(f"Invalid IRI triplet: head={positive_sample[0]}, tail={positive_sample[2]}, n_items={self.n_items}")
        
        negative_sample = torch.LongTensor(negative_sample)
        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample, subsampling_weight, self.mode

    @staticmethod
    def count_frequency(triplets, start=4):
        count = {}
        for head, relation, tail in triplets:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1
        return count

    @staticmethod
    def get_true_item_and_tag(triples):
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

        return true_head, true_tail

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode

class PretrainDataIterator(object):
    def __init__(self, pre_IRT_dataloader_neg_item,
        pre_IRT_dataloader_neg_tag,
        pre_TRT_dataloader, pre_IRI_dataloader, IRT_ratio, TRT_ratio, IRI_ratio):
        self.iterator_IRT_item = self.one_shot_iterator(pre_IRT_dataloader_neg_item)
        self.iterator_IRT_tag = self.one_shot_iterator(pre_IRT_dataloader_neg_tag)
        self.iterator_TRT = self.one_shot_iterator(pre_TRT_dataloader)
        self.iterator_IRI = self.one_shot_iterator(pre_IRI_dataloader)
        self.ratio_list = int(IRT_ratio*1000)*['IRT'] + int(TRT_ratio*1000)*['TRT'] + int(IRI_ratio*1000)*['IRI']
        self.list_len = len(self.ratio_list)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        data_type = self.ratio_list[random.randint(0, self.list_len-1)]
        if data_type == 'IRT':
            if self.step % 2 == 0:
                data = next(self.iterator_IRT_item)
            else:
                data = next(self.iterator_IRT_tag)
        elif data_type == 'TRT':
            data = next(self.iterator_TRT)
        else:
            data = next(self.iterator_IRI)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data

class TestforPreTrainDataset(Dataset):
    def __init__(self, triplets, triplets_test, n_params, mode, args):
        self.triplets = triplets
        self.n_tag = n_params['n_tags']
        self.n_item = n_params['n_items']
        self.n_entity = n_params['n_entities']
        self.mode = mode

        self.true_tag = self.get_true_item_and_tag(self.triplets)
        
        random.seed(args.seed)
        self.data = triplets_test
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        item, relation, tag = self.data[idx]

        # メモリ効率改善: リスト内包表記からnumpy配列操作に変更
        n_tags = self.n_entity - self.n_item
        tag_idx = tag - self.n_item  # 正解タグのインデックス
        negative_sample = np.arange(n_tags, dtype=np.int64)  # 0からn_tags-1
        filter_bias = np.zeros(n_tags, dtype=np.float32)  # デフォルト0
        
        # 訓練データに含まれるタグにはfilter_biasを-1に、negative_sampleを正解タグIDに
        # （元の実装: else (-1, tag) の再現）
        true_tags = self.true_tag.get((item, relation), np.array([], dtype=np.int64))
        if len(true_tags) > 0:
            true_tag_indices = true_tags - self.n_item
            # 範囲内のインデックスのみ使用
            valid_mask = (true_tag_indices >= 0) & (true_tag_indices < n_tags)
            filter_bias[true_tag_indices[valid_mask]] = -1.0
            negative_sample[true_tag_indices[valid_mask]] = tag_idx  # 元の実装と同じ
        
        # 正解タグはfilter_bias=0、negative_sample=自身のインデックス（評価対象）
        # 境界チェック: tag_idxが有効な範囲内であることを確認
        if 0 <= tag_idx < n_tags:
            filter_bias[tag_idx] = 0.0
            negative_sample[tag_idx] = tag_idx

        positive_sample = torch.LongTensor((item, relation, tag_idx))
        negative_sample = torch.from_numpy(negative_sample)
        filter_bias = torch.from_numpy(filter_bias)

        return positive_sample, negative_sample, filter_bias, self.mode
        
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode

    @staticmethod
    def get_true_item_and_tag(triples):
        true_tag = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tag:
                true_tag[(head, relation)] = []
            true_tag[(head, relation)].append(tail)

        for head, relation in true_tag:
            true_tag[(head, relation)] = np.array(list(set(true_tag[(head, relation)])))                 

        return true_tag


class PreTrainDataset_inter(Dataset):
    def __init__(self, item_tag, data, negative_sample_size, n_params):
        self.item_tag = item_tag
        
        self.data = data
        self.len = len(self.data)

        self.negative_sample_size = negative_sample_size
        self.n_items = n_params['n_items']
        self.n_tags = n_params['n_tags']
        self.n_relation = n_params['n_relations']

        self.count_item = self.count_frequency(self.item_tag)

        self.tureitem = self.get_ture_item(self.item_tag)

        self.padding_length = self.count_length(self.item_tag)
        self.add_padding()
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_sample, rel_tags = self.data[idx]
        rel_tags = torch.tensor(rel_tags)

        relations = rel_tags[:, 0]
        tags = rel_tags[:, 1]

        ori_tags = "".join(str(rel_tag[1]) for rel_tag in self.item_tag[positive_sample])

        subsampling_weight = self.count_item[positive_sample]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.n_items, size=self.negative_sample_size*2)
            mask = np.in1d(
                negative_sample, 
                self.tureitem[ori_tags],
                assume_unique=True, 
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.LongTensor(negative_sample)
        positive_sample = torch.LongTensor([positive_sample])
        tags = tags-self.n_items
        
        return positive_sample, negative_sample, relations, tags, subsampling_weight
    
    @staticmethod
    def get_ture_item(item_tag):
        true_item = defaultdict(list)
        for item, rel_tags in item_tag.items():
            true_item["".join(str(i[1]) for i in rel_tags)].append(item)
        return true_item

    @staticmethod
    def count_frequency(item_tag):
        count_item = {}

        for item, rel_tags in item_tag.items():
            if item not in count_item:
                count_item[item] = len(rel_tags) + 1
            else:
                raise ValueError("Repeated item, please check the readdata.py")
            
        return count_item

    @staticmethod
    def count_length(item_tag):
        length_list = [len(rel_tags) for rel_tags in item_tag.values()]
        length_list.sort()
        return length_list[math.ceil(len(length_list)*0.9)]

    def add_padding(self):
        for i, (item, rel_tags) in enumerate(self.data):
            if len(rel_tags) < self.padding_length:
                rel_tags.extend([[self.n_relation, self.n_tags+self.n_items]] * (self.padding_length - len(rel_tags)))
            elif len(rel_tags) > self.padding_length:
                self.data[i][1] = rel_tags[:self.padding_length]

class DataLoaderIterator(object):
    
    def __init__(self, dataloader):
        
        self.dataloader = self.one_shot_iterator(dataloader)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        data = next(self.dataloader)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data

class TestforPreTrainInterDataset(Dataset):
    def __init__(self, item_tag, data, n_params, args):

        self.item_tag = item_tag
        self.data= data
       

        self.n_items = n_params['n_items']
        self.n_tags = n_params['n_tags']
        self.n_relation = n_params['n_relations']

        self.tureitem = self.get_ture_item(self.item_tag)
        self.padding_length = self.count_length(self.item_tag)
        self.add_padding()

        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        item, rel_tags = self.data[idx]
        rel_tags = torch.tensor(rel_tags)
        relations = rel_tags[:, 0]
        tags = rel_tags[:, 1]
        positive_sample = item
        ori_tags = "".join(str(rel_tag[1]) for rel_tag in self.item_tag[positive_sample])

        # メモリ効率改善: リスト内包表記からnumpy配列操作に変更
        negative_sample = np.arange(self.n_items, dtype=np.int64)
        filter_bias = np.zeros(self.n_items, dtype=np.float32)
        
        # 同じタグを持つアイテムにはfilter_biasを-1、negative_sampleをitemに
        true_items = self.tureitem.get(ori_tags, [])
        if len(true_items) > 0:
            true_items_arr = np.array(true_items, dtype=np.int64)
            valid_mask = (true_items_arr >= 0) & (true_items_arr < self.n_items)
            valid_items = true_items_arr[valid_mask]
            filter_bias[valid_items] = -1.0
            negative_sample[valid_items] = item
        
        # 正解アイテムはfilter_bias=0（評価対象）
        # 境界チェック: itemが有効な範囲内であることを確認
        if 0 <= item < self.n_items:
            filter_bias[item] = 0.0
            negative_sample[item] = item
        
        negative_sample = torch.from_numpy(negative_sample)
        filter_bias = torch.from_numpy(filter_bias)
        
        tags = tags - self.n_items
        positive_sample = torch.LongTensor([positive_sample])
        
        return positive_sample, negative_sample, relations, tags, filter_bias

    @staticmethod
    def get_ture_item(item_tag):
        true_item = defaultdict(list)
        for item, rel_tags in item_tag.items():
            true_item["".join(str(i[1]) for i in rel_tags)].append(item)
            
        return true_item

    @staticmethod
    def count_length(item_tag):
        length_list = [len(rel_tags) for rel_tags in item_tag.values()]
        length_list.sort()
        return length_list[math.ceil(len(length_list)*0.9)]

    def add_padding(self):
        for i, (item, rel_tags) in enumerate(self.data):
            if len(rel_tags) < self.padding_length:
                rel_tags.extend([[self.n_relation, self.n_tags+self.n_items]] * (self.padding_length - len(rel_tags)))
            elif len(rel_tags) > self.padding_length:
                self.data[i][1] = rel_tags[:self.padding_length]



class TrainDataset(Dataset):
    def __init__(self, ui_pair, ui_set, item_tag, args, n_params):
        def to_set(dict):
            for k, v in dict.items():
                dict[k] = set(v)
            return dict

        self.item_tag = item_tag
        
        self.ui_pair = ui_pair
        self.ui_set = ui_set
        self.negative_sample_size = args.train_negative_sample_size
        self.n_items = n_params['n_items']
        self.n_tags = n_params['n_tags']
        self.n_relation = n_params['n_relations']
        
        self.count_user, self.count_item = self.count_frequency(self.ui_pair)
        self.len = len(ui_pair)

        self.item_padding_length = self.count_item_length(self.ui_set)
        self.tag_padding_length = self.count_tag_length(self.item_tag)
        self.add_padding()

        self.item2tags()

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        user, positive_sample = self.ui_pair[idx]
        items = self.ui_set[user].copy()
        if positive_sample in items:
            items.remove(positive_sample)
        else:
            items = items[:-1]

        subsampling_weight = self.count_user[user] + self.count_item[positive_sample]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.n_items, size=self.negative_sample_size*2)
            mask = np.in1d(
                negative_sample, 
                self.ui_set[user], 
                assume_unique=True, 
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.LongTensor(negative_sample)
        positive_sample = torch.LongTensor([positive_sample])

        items = torch.LongTensor(items)

        user = torch.LongTensor([user])
        relations = self.relations[items]
        tags = self.tags[items]
        relations = torch.LongTensor(relations)
        tags = torch.LongTensor(tags)-self.n_items
        
        return user, items, relations, tags, positive_sample, negative_sample, subsampling_weight

    @staticmethod
    def count_frequency(uipair, start=4):
        count_user = {}
        count_item = {}

        for user, item in uipair:
            if user not in count_user:
                count_user[user] = start
            else:
                count_user[user] += 1
            if item not in count_item:
                count_item[item] = start
            else:
                count_item[item] += 1
            
        return count_user, count_item
    
    @staticmethod
    def count_item_length(ui_set):
        length_list = [len(items)-1 for items in ui_set.values()]
        length_list.sort()
        return length_list[int(len(length_list)*0.85)+1]

    @staticmethod
    def count_tag_length(item_tag):
        length_list = [len(rel_tags) for rel_tags in item_tag.values()]
        length_list.sort()
        return length_list[int(len(length_list)*0.9)]

    def add_padding(self):
        for user, item_list in self.ui_set.items():
            if len(item_list) < self.item_padding_length:
                item_list.extend([self.n_items] * (self.item_padding_length - len(item_list)))
            if len(item_list) > self.item_padding_length:
                self.ui_set[user] = item_list[-self.item_padding_length:]
        for item, rel_tags in self.item_tag.items():
            if len(rel_tags) < self.tag_padding_length:
                rel_tags.extend([[self.n_relation, self.n_tags+self.n_items]] * (self.tag_padding_length - len(rel_tags)))
            elif len(rel_tags) > self.tag_padding_length:
                self.item_tag[item] = rel_tags[:self.tag_padding_length]
    
    def item2tags(self):
        self.relations = torch.zeros(self.n_items+1, self.tag_padding_length)
        self.tags = torch.zeros(self.n_items+1, self.tag_padding_length)
        self.relations[:, :] = self.n_relation
        self.relations = self.relations.long()
        self.tags[:, :] = self.n_tags
        self.tags = self.tags.long()

        for item, rel_tags in self.item_tag.items():
            rel_tags = torch.tensor(rel_tags)
            self.relations[item] = rel_tags[:, 0].squeeze()
            self.tags[item] = rel_tags[:, 1].squeeze()

class TestDataset(Dataset):
    def __init__(self, train_ui_set, test_ui_set, item_tag, n_params):
        def to_set(dict):
            for k, v in dict.items():
                dict[k] = set(v)
            return dict

        self.train_ui_set = train_ui_set
        self.test_ui_set = []
        for k, v in test_ui_set.items():
            self.test_ui_set.append([k, v])
        self.item_tag = item_tag

        self.n_items = n_params['n_items']
        self.n_tags = n_params['n_tags']
        self.n_relation = n_params['n_relations']
        self.len = len(test_ui_set)

        self.item_padding_length = self.count_item_length(self.train_ui_set)
        self.tag_padding_length = self.count_tag_length(self.item_tag)
        self.add_padding()

        self.item2tags()

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        user, test_items = self.test_ui_set[idx]

        train_items = self.train_ui_set[user]

        # メモリ効率改善: リスト内包表記からnumpy配列操作に変更
        negative_sample = np.arange(self.n_items, dtype=np.int64)
        filter_bias = np.zeros(self.n_items, dtype=np.float32)
        
        # 訓練データに含まれるアイテムにはfilter_biasを-1、negative_sampleをn_itemsに
        train_items_arr = np.array(train_items, dtype=np.int64)
        valid_mask = (train_items_arr >= 0) & (train_items_arr < self.n_items)
        valid_items = train_items_arr[valid_mask]
        filter_bias[valid_items] = -1.0
        negative_sample[valid_items] = self.n_items
        
        positive_sample = torch.LongTensor([self.n_items])
        negative_sample = torch.from_numpy(negative_sample)
        filter_bias = torch.from_numpy(filter_bias)

        train_items = torch.LongTensor(train_items)

        relations = self.relations[train_items]
        tags = self.tags[train_items]
        relations = torch.LongTensor(relations)
        tags = torch.LongTensor(tags)-self.n_items

        return user, train_items, relations, tags, positive_sample, negative_sample, filter_bias
    
    @staticmethod
    def count_item_length(ui_set):
        length_list = [len(items) for items in ui_set.values()]
        length_list.sort()
        return length_list[int(len(length_list)*0.85)]

    @staticmethod
    def count_tag_length(item_tag):
        length_list = [len(rel_tags) for rel_tags in item_tag.values()]
        length_list.sort()
        return length_list[int(len(length_list)*0.9)]

    def add_padding(self):
        for user, item_list in self.train_ui_set.items():
            if len(item_list) < self.item_padding_length:
                item_list.extend([self.n_items] * (self.item_padding_length - len(item_list)))
            if len(item_list) > self.item_padding_length:
                self.train_ui_set[user] = item_list[-self.item_padding_length:]
        for item, rel_tags in self.item_tag.items():
            if len(rel_tags) < self.tag_padding_length:
                rel_tags.extend([[self.n_relation, self.n_tags+self.n_items]] * (self.tag_padding_length - len(rel_tags)))
            elif len(rel_tags) > self.tag_padding_length:
                self.item_tag[item] = rel_tags[:self.tag_padding_length]

    def item2tags(self):
        self.relations = torch.zeros(self.n_items+1, self.tag_padding_length)
        self.tags = torch.zeros(self.n_items+1, self.tag_padding_length)
        self.relations[:, :] = self.n_relation
        self.relations = self.relations.long()
        self.tags[:, :] = self.n_tags
        self.tags = self.tags.long()

        for item, rel_tags in self.item_tag.items():
            rel_tags = torch.tensor(rel_tags)
            self.relations[item] = rel_tags[:, 0].squeeze()
            self.tags[item] = rel_tags[:, 1].squeeze()