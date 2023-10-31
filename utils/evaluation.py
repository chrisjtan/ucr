import sys
import os
import copy
import torch
import random
import numpy as np
import tqdm
from collections import defaultdict
from multiprocessing import Process, Queue


# evaluate on val set
def evaluate_valid(model, dataset, args):
    train = dataset.train_dict
    valid = dataset.val_dict
    future = dataset.future_dict
    test = dataset.test_dict

    usernum = dataset.user_num
    itemnum = dataset.item_num
    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum > 40000:
        users = np.load(os.path.join("datasets", args.dataset+"_test_users.pickle"), allow_pickle=True)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1:
            raise ValueError('train or valid empty!')

        seq = np.zeros([args.max_seq], dtype=np.int32)
        idx = args.max_seq - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        for i in future[u]:
            rated.add(i)
        item_idx = [valid[u][0]]
        for i in range(itemnum):
            if ((i+1) != test[u][0]) and ((i+1) not in rated):
                item_idx.append((i+1))

        input_user = torch.Tensor([u])
        input_seq = torch.LongTensor([seq]).to(args.device)
        input_tar = torch.LongTensor(item_idx).to(args.device)
        predictions = -model.predict(input_user, input_seq, input_tar)

        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user

# # TODO: merge evaluate functions for test and val set
# # evaluate on test set
def evaluate(model, dataset, args):
    train = dataset.train_dict
    valid = dataset.val_dict
    future = dataset.future_dict
    test = dataset.test_dict


    usernum = dataset.user_num
    itemnum = dataset.item_num

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>40000:
        # users = random.sample(range(1, usernum + 1), 10000)
        users = np.load(os.path.join("datasets", args.dataset +"_test_users.pickle"), allow_pickle=True)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.max_seq], dtype=np.int32)
        idx = args.max_seq - 1

        # process future data
        for i in reversed(future[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        rated.add(valid[u][0])
        for i in future[u]:
            rated.add(i)
        item_idx = [test[u][0]]
        for i in range(itemnum):
            if ((i+1) != test[u][0]) and ((i+1) not in rated):
                item_idx.append((i+1))
        input_user = torch.Tensor([u])
        input_seq = torch.LongTensor([seq]).to(args.device)
        input_tar = torch.LongTensor(item_idx).to(args.device)
        predictions = -model.predict(input_user, input_seq, input_tar)
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def generate_pool(dataset, pool_size):
    test_pool = {}
    if pool_size == -1:
        pool_size = dataset.item_num
    train = dataset.train_dict
    valid = dataset.val_dict
    future = dataset.future_dict
    test = dataset.test_dict
    usernum = dataset.user_num
    itemnum = dataset.item_num
    for u in tqdm.trange(1, usernum + 1):
        rated = set(train[u])
        rated.add(0)
        rated.add(valid[u][0])
        item_idx = [test[u][0]]
        for i in range(itemnum):
            if ((i+1) != test[u][0]) and ((i+1) not in rated):
                item_idx.append((i+1))
        if len(item_idx) > pool_size:
            item_idx = random.sample(item_idx, pool_size)
        test_pool[u] = item_idx
    return test_pool