import numpy as np
import pandas as pd
import torch
from collections import Counter


def load_foursquare_data(dataset_dir):
    print('Loading original data ......')
    pois = pd.read_csv(dataset_dir, sep=' ')

    all_user_pois = [[i for i in upois.split('/')] for upois in pois['u_pois']]
    all_users = [i for i in pois['u_id']]
    all_train_data = [item for upois in all_user_pois for item in upois]
    all_items = set(all_train_data)

    user_num, item_num = len(all_users), len(set(all_train_data))
    print('\tusers, items:  = {v1}, {v2}'.format(v1=user_num, v2=item_num))

    # 0-n的重新映射
    item_aliases_dict = dict(zip(all_items, range(item_num)))
    user_aliases_dict = dict(zip(all_users, range(user_num)))

    # 映射为Index
    item_aliases_list = [[item_aliases_dict[i] for i in item] for item in all_user_pois]
    user_aliases_list = [user_aliases_dict[i] for i in all_users]

    print('\tsplit train data set and test data set ...')

    train_set = torch.zeros([user_num, item_num])
    test_set = dict()
    train_set_item = dict()

    for u in user_aliases_list:
        the_user_item_list = item_aliases_list[u]

        # 切割，将最后一个作为测试数据
        tran_the_user = the_user_item_list[:-1]
        test_the_user = the_user_item_list[-1]

        the_user_item_counter = Counter(tran_the_user)
        train_set_item[u] = set(tran_the_user)

        for item in tran_the_user:
            train_set[u, item] = the_user_item_counter[item]

        test_set[u] = test_the_user

    return user_num, item_num, \
            item_aliases_dict, user_aliases_dict, \
            train_set, test_set, train_set_item
