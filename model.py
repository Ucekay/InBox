#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from utils.eval_tuple import eval_tuple
from einops import rearrange, reduce, repeat

def Identity(x):
    return x

class BoxOffsetIntersection(nn.Module):
    
    def __init__(self, dim):
        super(BoxOffsetIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings))
        layer1_mean = torch.mean(layer1_act, dim=1) 
        gate = torch.sigmoid(self.layer2(layer1_mean))
        offset, _ = torch.min(embeddings, dim=1)

        return offset * gate

class CenterIntersection(nn.Module):

    def __init__(self, dim):
        super(CenterIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings))
        attention = F.softmax(self.layer2(layer1_act), dim=1)
        embedding = torch.sum(attention * embeddings, dim=1).unsqueeze(1)

        return embedding

class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim = dim
        self.Q = nn.Linear(self.dim, self.dim, bias=False)
        self.K = nn.Linear(self.dim, self.dim, bias=False)
        self.V = nn.Linear(self.dim, self.dim, bias=False)

        nn.init.xavier_uniform_(self.Q.weight)
        nn.init.xavier_uniform_(self.K.weight)
        nn.init.xavier_uniform_(self.V.weight)

    def forward(self, embeddings):
        query = self.Q(embeddings)
        key = self.K(embeddings)
        value = self.V(embeddings)
        
        key_trans = torch.transpose(key, -2, -1)

        attn = torch.matmul(query, key_trans) / torch.sqrt(torch.tensor([self.dim]).cuda())
        attn = torch.where(attn==0, torch.tensor([1e-10]).cuda(), attn)
        attn = torch.softmax(attn, dim=-1)
        embeddings = torch.matmul(attn, value)
    
        return embeddings

class TagBoxAttnInter(nn.Module):
    def __init__(self, dim):
        super(TagBoxAttnInter, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim*2, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings, item):
        item = repeat(item, 'b ip d -> b ip tp d', tp=embeddings.size()[-2])
        input = rearrange([embeddings, item], 'n b ip tp d -> b ip tp (n d)')

        layer1_act = F.relu(self.layer1(input))
        attention = F.softmax(self.layer2(layer1_act), dim=-1)
        embedding = torch.sum(attention * embeddings, dim=-2)

        return embedding

class InterestBoxAggerator(nn.Module):
    def __init__(self, dim, interest_num):
        super(InterestBoxAggerator, self).__init__()
        self.dim = dim
        self.interest_num = interest_num
        self.layer1 = nn.Linear(self.dim, self.dim, bias=False)
        self.layer2 = nn.Linear(self.dim, self.interest_num, bias=False)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
    
    def forward(self, embeddings):
        tmp = self.layer2(F.relu(self.layer1(embeddings)))
        weights = torch.softmax(tmp, dim=1)
        embeddings = torch.matmul(weights.transpose(-1, -2), embeddings)
        return embeddings


class Model(nn.Module):
    def __init__ (self, args, n_params):
        super(Model, self).__init__()
        self.dim = args.dim
        self.nitem = n_params['n_items']
        self.ntag = n_params['n_tags']
        self.nentity = n_params['n_entities']
        self.nrelation = n_params['n_relations']
        self.nuser = n_params['n_users']
        self.ninterest = args.interest_num
        self.args = args

        self.gamma = nn.Parameter(
            torch.Tensor([args.gamma]), 
            requires_grad=False
        )
        self.epsilon = 2.0
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / self.dim]), 
            requires_grad=False
        )

        activation, cen = eval_tuple(args.box_mode)
        self.cen = cen
        if activation == 'none':
            self.func = Identity
        elif activation == 'relu':
            self.func = F.relu
        elif activation == 'softplus':
            self.func = F.softplus
        elif activation == 'abs1':
            self.func = torch.abs

        self.entity_dim = self.dim
        self.tag_dim = self.dim
        self.relation_dim = self.dim

        self.user_embedding = nn.Embedding(self.nuser, self.entity_dim)
        nn.init.uniform_(
            tensor=self.user_embedding.weight, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )

        self.item_embedding = nn.Embedding(self.nitem+1, self.entity_dim, padding_idx=self.nitem)
        nn.init.uniform_(
            tensor=self.item_embedding.weight[:-1], 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )

        self.tag_center_embedding = nn.Embedding(self.ntag+1, self.tag_dim, padding_idx=self.ntag)
        nn.init.uniform_(
            tensor=self.tag_center_embedding.weight[:-1], 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        self.tag_offset_embedding = nn.Embedding(self.ntag+1, self.tag_dim, padding_idx=self.ntag)
        nn.init.uniform_(
                tensor=self.tag_offset_embedding.weight[:-1], 
                a=0., 
                b=self.embedding_range.item()
            )
        self.interest_center_embedding = nn.Embedding(self.nitem+1, self.tag_dim, padding_idx=self.nitem)
        nn.init.uniform_(
            tensor=self.interest_center_embedding.weight[:-1], 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        self.interest_offset_embedding = nn.Embedding(self.nitem+1, self.tag_dim, padding_idx=self.nitem)
        nn.init.uniform_(
            tensor=self.interest_offset_embedding.weight[:-1], 
            a=0., 
            b=self.embedding_range.item()
        )

        self.relation_center_embedding = nn.Embedding(self.nrelation+1, self.relation_dim, padding_idx=self.nrelation)
        nn.init.uniform_(
            tensor=self.relation_center_embedding.weight[:-1], 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        self.relation_offset_embedding = nn.Embedding(self.nrelation+1, self.entity_dim, padding_idx=self.nrelation)
        nn.init.uniform_(
            tensor=self.relation_offset_embedding.weight[:-1], 
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        
        self.tag_center_attention = Attention(self.entity_dim)
        self.tag_offset_attention = Attention(self.entity_dim)

        self.tag_center_net = CenterIntersection(self.entity_dim)
        self.tag_offset_net = BoxOffsetIntersection(self.entity_dim)

        self.tag_center_attn_inter = TagBoxAttnInter(self.dim)
        self.tag_offset_attn_inter = TagBoxAttnInter(self.dim)

        self.interest_center_attention = Attention(self.entity_dim)
        self.interest_offset_attention = Attention(self.entity_dim)

        self.interest_center_net = InterestBoxAggerator(self.entity_dim, self.ninterest)
        self.interest_offset_net = InterestBoxAggerator(self.entity_dim, self.ninterest)
    
    def boxes_base_inter (self, center_embeddings, offset_embeddings):
        max_points = center_embeddings + offset_embeddings
        min_points = center_embeddings - offset_embeddings

        max_point = torch.min(max_points, dim = -2).values
        min_point = torch.max(min_points, dim = -2).values

        center_embedding = (max_point + min_point)/2
        offset_embedding = self.func(max_point - min_point)

        return center_embedding.unsqueeze(1), offset_embedding.unsqueeze(1)

    def point_box_logit (self, point, box):
        box_center_embedding, box_offset_embedding = box
        delta = (point - box_center_embedding).abs()
        distance_out = self.func(delta - box_offset_embedding)
        distance_in = torch.min(delta, box_offset_embedding)

        logit = self.gamma - torch.norm(distance_out, p=1, dim=-1) - self.cen * torch.norm(distance_in, p=1, dim=-1)

        return logit

    def box_box_logit (self, box1, box2, relation):
        box1_center, box1_offset = box1
        box2_center, box2_offset = box2
        relation_center, relation_offset = relation

        distance_center = box1_center - (box2_center + relation_center)
        distance_offset = box1_offset - (box2_offset + relation_offset)
        logit_center = self.gamma - torch.norm(distance_center, p=1, dim=-1)
        logit_offset = self.gamma - torch.norm(distance_offset, p=1, dim=-1)

        return (logit_center+logit_offset)/2

    def point_point_logit (self, point1, point2, relation):
        distance = point1 - (point2 + relation)
        logit = self.gamma - torch.norm(distance, p=1, dim=-1)
        return logit
    
    def point_point_logit2 (self, point1, point2):
        distance = point1 - point2
        logit = self.gamma - torch.norm(distance, p=1, dim=-1)
        return logit

    def forward (self, sample, mode, flag='train'):
        if mode in ['IRT-item', 'IRT-tag', 'IRI', 'TRT']:
            positive_logit, negative_logit = self.forward_pretrain(sample, mode)
        elif mode == 'pretrain_inter':
            positive_logit, negative_logit = self.forward_pretrain_inter(sample)
        elif mode == 'train':
            positive_logit, negative_logit = self.forward_recommender(sample, flag)
        else:
            raise ValueError('mode %s not supported' % mode)
        
        return positive_logit, negative_logit

    def forward_pretrain (self, sample, mode):
        positive_sample, negative_sample = sample
        if self.args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()

        if mode == 'IRT-item':
            item = self.item_embedding(positive_sample[:,0]).unsqueeze(1)
            relation_center = self.relation_center_embedding(positive_sample[:,1]).unsqueeze(1)
            relation_offset = self.relation_offset_embedding(positive_sample[:,1]).unsqueeze(1)
            tag_center = self.tag_center_embedding(positive_sample[:,2]).unsqueeze(1)
            tag_offset = self.func(self.tag_offset_embedding(positive_sample[:,2]).unsqueeze(1))
            item_neg = self.item_embedding(negative_sample)

            tag_center = tag_center + relation_center
            tag_offset = self.func(tag_offset + relation_offset)

            positive_logit = self.point_box_logit(item, (tag_center, tag_offset))
            negative_logit = self.point_box_logit(item_neg, (tag_center, tag_offset))
        
        elif mode == 'IRT-tag':
            item = self.item_embedding(positive_sample[:,0]).unsqueeze(1)
            relation_center = self.relation_center_embedding(positive_sample[:,1]).unsqueeze(1)
            relation_offset = self.relation_offset_embedding(positive_sample[:,1]).unsqueeze(1)
            tag_center = self.tag_center_embedding(positive_sample[:,2]).unsqueeze(1)
            tag_offset = self.func(self.tag_offset_embedding(positive_sample[:,2]).unsqueeze(1))

            tag_center_neg = self.tag_center_embedding(negative_sample)
            tag_offset_neg = self.func(self.tag_offset_embedding(negative_sample))

            tag_center = tag_center + relation_center
            tag_offset = self.func(tag_offset + relation_offset)
            tag_center_neg = tag_center_neg + relation_center
            tag_offset_neg = self.func(tag_center_neg + relation_offset)

            positive_logit = self.point_box_logit(item, (tag_center, tag_offset))
            negative_logit = self.point_box_logit(item, (tag_center_neg, tag_offset_neg))

        elif mode == 'IRI':
            head = self.item_embedding(positive_sample[:,0]).unsqueeze(1)
            relation = self.relation_center_embedding(positive_sample[:,1]).unsqueeze(1)
            tail = self.item_embedding(positive_sample[:,2]).unsqueeze(1)

            tail_neg = self.item_embedding(negative_sample)

            positive_logit = self.point_point_logit(head, tail, relation)
            negative_logit = self.point_point_logit(head, tail_neg, relation)

        else:
            head_center = self.tag_center_embedding(positive_sample[:,0]).unsqueeze(1)
            head_offset = self.func(self.tag_offset_embedding(positive_sample[:,0]).unsqueeze(1))
            relation_center = self.relation_center_embedding(positive_sample[:,1]).unsqueeze(1)
            relation_offset = self.relation_offset_embedding(positive_sample[:,1]).unsqueeze(1)
            tail_center = self.tag_center_embedding(positive_sample[:,2]).unsqueeze(1)
            tail_offset = self.func(self.tag_offset_embedding(positive_sample[:,2]).unsqueeze(1))

            tail_center_neg = self.tag_center_embedding(negative_sample)
            tail_offset_neg = self.func(self.tag_offset_embedding(negative_sample))

            positive_logit = self.box_box_logit((head_center, head_offset), (tail_center, tail_offset), (relation_center, relation_offset))
            negative_logit = self.box_box_logit((head_center, head_offset), (tail_center_neg, tail_offset_neg), (relation_center, relation_offset))  

        return positive_logit, negative_logit

    def forward_pretrain_inter (self, sample):
        positive_sample, negative_sample, relations, tags = sample
        if self.args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            relations = relations.cuda()
            tags = tags.cuda()

        positive_item = self.item_embedding(positive_sample.squeeze()).unsqueeze(1)
        negative_item = self.item_embedding(negative_sample)
        relations_center = self.relation_center_embedding(relations)
        relations_offset = self.relation_offset_embedding(relations)
        tags_center = self.tag_center_embedding(tags) + relations_center
        tags_offset = self.func(self.func(self.tag_offset_embedding(tags)) + relations_offset)

        # neural network intersection
        interest_center = self.tag_center_net(tags_center)
        interest_offset = self.func(self.tag_offset_net(tags_offset)).unsqueeze(1)
        # M-M intersection
        # interest_center, interest_offset = self.boxes_base_inter(tags_center, tags_offset)

        positive_logit = self.point_box_logit(positive_item, (interest_center, interest_offset))
        negative_logit = self.point_box_logit(negative_item, (interest_center, interest_offset))
        
        with torch.no_grad():
            self.interest_center_embedding.weight.requires_grad_ = False
            self.interest_offset_embedding.weight.requires_grad_ = False
            self.interest_center_embedding.weight[positive_sample] = interest_center
            self.interest_offset_embedding.weight[positive_sample] = interest_offset
            self.interest_center_embedding.weight.requires_grad_ = True
            self.interest_offset_embedding.weight.requires_grad_ = True

        return positive_logit, negative_logit

    def forward_recommender (self, sample, flag):
        user, items, relations, tags, positive_sample, negative_sample = sample
        if self.args.cuda:
            user = user.cuda()
            items = items.cuda()
            relations = relations.cuda()
            tags = tags.cuda()
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
        
    
        positive_item = self.item_embedding(positive_sample.squeeze()).unsqueeze(1)
        negative_item = self.item_embedding(negative_sample)

        
        base_interests_center = self.interest_center_embedding(items)
        base_interests_offset = self.func(self.interest_offset_embedding(items))
        
        bias = self.item_embedding(items)
        relations_center = self.relation_center_embedding(relations)
        relations_offset = self.relation_offset_embedding(relations)
        tags_center = self.tag_center_embedding(tags) + relations_center
        tags_offset = self.func(self.func(self.tag_offset_embedding(tags)) + relations_offset)
        
        attn_interests_center = self.tag_center_attn_inter(tags_center, bias)
        attn_interests_offset = self.tag_offset_attn_inter(tags_offset, bias)

        
        interests_center = (base_interests_center + attn_interests_center) / 2
        interests_offset = (base_interests_offset + attn_interests_offset) / 2

        num = base_interests_center.sum(-1).bool().int().sum(-1).unsqueeze(-1).unsqueeze(-1)
        num = torch.where(num==0, torch.tensor([1]).cuda(), num)
        
        user_center = interests_center.sum(dim=1).unsqueeze(1)/num
        user_offset = interests_offset.sum(dim=1).unsqueeze(1)/num

        user_positive_logit = self.point_box_logit(positive_item, (user_center, user_offset))
        user_negative_logit = self.point_box_logit(negative_item, (user_center, user_offset))

        return user_positive_logit, user_negative_logit

    @staticmethod
    def train_step (model, optimizer, train_iterator, args, train_mode):
        model.train()
        optimizer.zero_grad()

        if train_mode == 'pretrain':
            positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)
            input = (positive_sample, negative_sample)
        elif train_mode == 'pretrain_inter':
            positive_sample, negative_sample, relations, tags, subsampling_weight = next(train_iterator)
            input = (positive_sample, negative_sample, relations, tags)
            mode = 'pretrain_inter'
        elif train_mode == 'train':
            user, items, relations, tags, positive_sample, negative_sample, subsampling_weight = next(train_iterator)
            input = (user, items, relations, tags, positive_sample, negative_sample)
            mode = 'train'
        else:
            assert False, "Wrong train mode."

        if args.cuda:
            subsampling_weight = subsampling_weight.cuda()
        
        positive_logit, negative_logit = model(input, mode)

        negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
        positive_score = F.logsigmoid(positive_logit)

        positive_sample_loss = - (subsampling_weight * positive_score).sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()
        positive_sample_loss /= subsampling_weight.sum()
        negative_sample_loss /= subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        loss.backward()
        optimizer.step()

        if train_mode == 'pretrain':
            log = {
                'pretrain_positive_sample_loss': positive_sample_loss.item(),
                'pretrain_negative_sample_loss': negative_sample_loss.item(),
                'loss': loss.item(),
            }
        elif train_mode == 'pretain_inter':
            log = {
                'pretrain_inter_positive_sample_loss': positive_sample_loss.item(),
                'pretrain_inter_negative_sample_loss': negative_sample_loss.item(),
                'loss': loss.item(),
            }
        else:
            log = {
                'train_positive_sample_loss': positive_sample_loss.item(),
                'train_negative_sample_loss': negative_sample_loss.item(),
                'loss': loss.item(),
            }
        
        return log
    
    @staticmethod
    def test_step (model, test_iterator, args, test_mode, test_user_set=None):

        def dcg_at_k(r, k, method=1):
            """Score is discounted cumulative gain (dcg)
            Relevance is positive real values.  Can use binary
            as the previous methods.
            Returns:
                Discounted cumulative gain
            """
            r = np.asarray(r, dtype=np.float64)[:k]
            if r.size:
                if method == 0:
                    return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
                elif method == 1:
                    return np.sum(r / np.log2(np.arange(2, r.size + 2)))
                else:
                    raise ValueError('method must be 0 or 1.')
            return 0.

        def ndcg_at_k(r, k, ground_truth, method=1):
            """Score is normalized discounted cumulative gain (ndcg)
            Relevance is positive real values.  Can use binary
            as the previous methods.
            Returns:
                Normalized discounted cumulative gain

                Low but correct defination
            """
            GT = set(ground_truth)
            if len(GT) > k :
                sent_list = [1.0] * k
            else:
                sent_list = [1.0]*len(GT) + [0.0]*(k-len(GT))
            dcg_max = dcg_at_k(sent_list, k, method)
            if not dcg_max:
                return 0.
            return dcg_at_k(r, k, method) / dcg_max 
    
        model.eval()
        
        logs = []
    
        with torch.no_grad():
            for data in tqdm(test_iterator):
                if test_mode == 'pretrain':
                    positive_sample, negative_sample, filter_bias, mode = data
                    input = (positive_sample, negative_sample)
                elif test_mode == 'pretrain_inter':
                    positive_sample, negative_sample, relations, tags, filter_bias = data
                    mode = 'pretrain_inter'
                    input = (positive_sample, negative_sample, relations, tags)
                elif test_mode == 'train':
                    user, items, relations, tags, positive_sample, negative_sample, filter_bias = data
                    mode = 'train'
                    input = (user, items, relations, tags, positive_sample, negative_sample)

                if args.cuda:
                    positive_sample = positive_sample.cuda()
                    filter_bias = filter_bias.cuda()

                batch_size = positive_sample.size(0)
                _, score = model(input, mode, flag='test')
                score = score/100 + filter_bias

                argsort = torch.argsort(score, dim = 1, descending=True)

                if test_mode == 'pretrain' or test_mode == 'pretrain_inter':
                    if test_mode == 'pretrain':
                        positive_arg = positive_sample[:, 2]
                    else:
                        positive_arg = positive_sample

                    for i in range(batch_size):
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1
                        ranking = 1 + ranking.item()
                        logs.append({
                            'MRR': 1.0/ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            'HITS@20': 1.0 if ranking <= 20 else 0.0
                        })
                else:
                    user = user.tolist()
                    argsort = argsort[:, :20].int().tolist()
                    for i in range(batch_size):
                        positive_item = test_user_set[user[i]]
                        flag_list = [1 if item in positive_item else 0 for item in argsort[i]]
                        num_positive = len(positive_item)
                        logs.append({
                            'recall@20': sum(flag_list) / num_positive,
                            'NDCG': ndcg_at_k(flag_list, 20, positive_item)
                        })
            
            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics
