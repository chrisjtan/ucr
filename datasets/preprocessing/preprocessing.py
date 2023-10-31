import os
import numpy as np
from collections import defaultdict

from sklearn.metrics import rand_score


class SeqDataset:
    def __init__(self, args, future_num=5):
        super().__init__()
        self.args = args
        self.future_num = future_num
        self.user_set = set()
        self.item_set = set()
        self.user_num = 0
        self.item_num = 0
        self.user_seq_dict = defaultdict(list)  # {'u': [i, i, i , ...], 'u':[i, i, i, ...]}
        self.train_dict = {}  # {'u': [i, i, i , ...], 'u':[i, i, i, ...]}
        self.val_dict = {}  # {'u': [i], 'u':[i]}
        self.future_dict = {}  # for prospective evaluation
        self.test_dict = {}  # {'u': [i], 'u':[i]}  leave one out
        self.preprocessing()

    def preprocessing(self):
        with open(os.path.join('datasets', self.args.dataset+'.txt')) as f:
            for line in f:
                u_i = line.rstrip().split(' ')
                user = int(u_i[0])
                item = int(u_i[1])
                self.user_set.add(user)
                self.item_set.add(item)
                self.user_seq_dict[user].append(item)

        self.user_num = len(self.user_set)
        self.item_num = len(self.item_set)

        for user, seq_items in self.user_seq_dict.items():
            if len(seq_items) < self.future_num + 3:
                # print("Find user with less than 3 history!")
                # pass
                self.train_dict[user] = seq_items
                self.val_dict[user] = []
                self.test_dict[user] = []
            else:
                self.train_dict[user] = seq_items[:-(2 + self.future_num)]
                self.val_dict[user] = [seq_items[-(2 + self.future_num)]]
                self.future_dict[user] = seq_items[-(1 + self.future_num):-1]
                self.test_dict[user] = [seq_items[-1]]



class SeqDatasetPoisoned():
    def __init__(self, args, p_r):
        super().__init__()
        self.args = args
        self.p_r = p_r
        self.user_set = set()
        self.item_set = set()
        self.user_num = 0
        self.item_num = 0
        self.user_seq_dict = defaultdict(list)  # {'u': [i, i, i , ...], 'u':[i, i, i, ...]}
        self.train_dict = {}  # {'u': [i, i, i , ...], 'u':[i, i, i, ...]}
        self.val_dict = {}  # {'u': [i], 'u':[i]}
        self.test_dict = {}  # {'u': [i], 'u':[i]}  leave one out
        self.future_dict = {}  # used for prospective evaluation
        self.preprocessing()
        self.generate_users()

    def preprocessing(self):
        with open(os.path.join('datasets', self.args.dataset+'.txt')) as f:
            for line in f:
                u_i = line.rstrip().split(' ')
                user = int(u_i[0])
                item = int(u_i[1])
                self.user_set.add(user)
                self.item_set.add(item)
                self.user_seq_dict[user].append(item)

        self.user_num = len(self.user_set)
        self.item_num = len(self.item_set)

        for user, seq_items in self.user_seq_dict.items():
            if len(seq_items) < 10:
                # print("Find user with less than 3 history!")
                # pass
                raise ValueError('seq length < 10!')
                self.train_dict[user] = seq_items
                self.val_dict[user] = []
                self.test_dict[user] = []
            else:
                self.train_dict[user] = seq_items[:-2]
                self.val_dict[user] = [seq_items[-2]]
                self.test_dict[user] = [seq_items[-1]]

    def generate_users(self):
        add_num = int(self.user_num * self.p_r)
        for i in range(add_num):  # iteratively add new user
            new_seq = []
            # generate sequence
            for j in range(self.args.seq_len):
                new_seq.append(np.random.randint(1, self.item_num + 1))
            new_user = self.user_num+1
            self.user_set.add(new_user)
            self.user_num += 1
            self.user_seq_dict[new_user].append(new_seq)
            self.train_dict[new_user] = new_seq
            self.val_dict[new_user] = []
            self.test_dict[new_user] = []


def data_init(args, future_num=5):
    """
    init dataset
    :param args:
    :param p_r: the ratio of adding poining data
    :return: dataset object
    """
    data_obj = SeqDataset(args, future_num)
    return data_obj
