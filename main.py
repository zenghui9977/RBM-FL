import numpy as np
import torch

from rbm import *
from load_foursquare_data import *
from federatedlearning import *
from utils import *

# 设置训练设备
CUDA = torch.cuda.is_available()
CUDA_DEVICE = 0
if CUDA:
    torch.cuda.set_device(CUDA_DEVICE)

# 定义文件名变量
DATA_FOLDER = 'data/'
FOURSQUARE_FILE = 'Foursquare.txt'
RESULT_FOLDER = 'result/5/'

user_num, item_num, \
item_aliases_dict, user_aliases_dict, \
train_set, test_set, train_set_item = load_foursquare_data(DATA_FOLDER + FOURSQUARE_FILE)

EPOCH = 5
BACTH_SIZE = 1
VISIBLE_UNIT = item_num
HIDDEN_UNIT = 128
CD_K = 4

params = Params(b=BACTH_SIZE, c=0.5, client_num=user_num, CUDA=CUDA,
                e=EPOCH, max_communication_round=40,
                rbm_visible_unit=VISIBLE_UNIT, rbm_hidden_unit= HIDDEN_UNIT, rbm_k=CD_K)
#
global_model_weights, global_model_visible_bias, global_model_hidden_bias, \
precision_mean, recall_mean, f_measure_mean, ndcg_mean, hit_num_mean= FederatedLearning(params, train_set, test_set, train_set_item)

# 需要保存模型参数，可以利用一下函数
# save_as_pt(RESULT_FOLDER, 'global_model_weights.pt', global_model_weights)
# save_as_pt(RESULT_FOLDER, 'global_model_visible_bias.pt', global_model_visible_bias)
# save_as_pt(RESULT_FOLDER, 'global_model_hidden_bias.pt', global_model_hidden_bias)

save_as_pkl(RESULT_FOLDER, 'precision', precision_mean)
save_as_pkl(RESULT_FOLDER, 'recall', recall_mean)
save_as_pkl(RESULT_FOLDER, 'f_measure', f_measure_mean)
save_as_pkl(RESULT_FOLDER, 'NDCG', ndcg_mean)
save_as_pkl(RESULT_FOLDER, 'hit_num', hit_num_mean)

#

# global_model = read_from_pt(RESULT_FOLDER, 'global_model.pt')
# test_rbm = RBM(params.rbm_visible_unit, params.rbm_hidden_unit, params.rbm_k, global_model, use_cuda=params.CUDA)

# rec_batch = 64
# train_data_loader =  torch.utils.data.DataLoader(train_set, batch_size = rec_batch)
#

# rec = torch.zeros([user_num, item_num])
# for i, batch in enumerate(train_data_loader):
#     if CUDA:
#         batch = batch.cuda()
#     pro = test_rbm.recommender(batch)
#
#     rec[i * rec_batch: i * rec_batch + len(batch)] = pro
#
#
# re_pro, re_index = rec.sort(1, descending=True)
# Top_K=100
# re_top_K_list = []
# for u in range(user_num):
#     u_train_set = set(train_set_item[u])
#     re_sort_index = re_index[u].cpu().numpy()
#     temp = []
#     for i in range(item_num):
#         if re_sort_index[i] not in u_train_set:
#             temp.append(re_sort_index[i])
#         if len(temp) == Top_K:
#             break
#     re_top_K_list.append(temp)

# 随机猜想
# re_top_K_rand_list = []
# for u in range(user_num):
#     u_train_set = set(train_set_item[u])
#     rec_rand = np.random.permutation(item_num)
#     temp = []
#     for i in range(item_num):
#         if rec_rand[i] not in u_train_set:
#             temp.append(rec_rand[i])
#         if len(temp) == Top_K:
#             break
#     re_top_K_rand_list.append(temp)


# save_as_pkl(RESULT_FOLDER, 'top@k_recommedation', re_top_K_list)
# save_as_npy(RESULT_FOLDER, 'ground_truth.npy', test_set)
# save_as_pkl(RESULT_FOLDER, 'rand_recommedation', re_top_K_rand_list)
#

# 读取保存的结果
# precision_list = read_from_pkl(RESULT_FOLDER, 'precision_4.pkl')
# recall_list = read_from_pkl(RESULT_FOLDER, 'recall_4.pkl')
# f_measure_list = read_from_pkl(RESULT_FOLDER, 'f_measure_4.pkl')
# ndcg_list = read_from_pkl(RESULT_FOLDER, 'NDCG_4.pkl')
# hit_num_list = read_from_pkl(RESULT_FOLDER, 'hit_num_4.pkl')


print('the statistical information in each federated learning:')
print('\t the best performance of top@[5, 10, 15, 20]')
print('\t \t \t \t round \t \t metrics value')
print('\t \t precision \t %s ---> %s' % (np.argmax(precision_mean, axis=0), max(precision_mean)))
print('\t \t recall \t %s ---> %s' % (np.argmax(recall_mean, axis=0), max(recall_mean)))
print('\t \t f_measure \t %s ---> %s' % (np.argmax(f_measure_mean, axis=0), max(f_measure_mean)))
print('\t \t NDCG \t \t %s ---> %s' % (np.argmax(ndcg_mean, axis=0), max(ndcg_mean)))
print('\t \t hit_num \t %s ---> %s' % (np.argmax(hit_num_mean, axis=0), max(hit_num_mean)))

# C = 0.15 round = 100
# C = 0.4 round = 100
# C = 0.5 round = 50

