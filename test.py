from load_foursquare_data import *
from utils import *
from rbm import *
from federatedlearning import init_global_model, evaluate_global_model


# 设置训练设备
CUDA = torch.cuda.is_available()
CUDA_DEVICE = 0
if CUDA:
    torch.cuda.set_device(CUDA_DEVICE)

# 定义文件名变量
DATA_FOLDER = 'data/'
FOURSQUARE_FILE = 'Foursquare.txt'
RESULT_FOLDER = 'result/rbm/'

user_num, item_num, \
item_aliases_dict, user_aliases_dict, \
train_set, test_set, train_set_item = load_foursquare_data(DATA_FOLDER + FOURSQUARE_FILE)

EPOCH = 200
BACTH_SIZE = 64
VISIBLE_UNIT = item_num
HIDDEN_UNIT = 128
CD_K = 4


weights, visible_bias, hidden_bias = init_global_model((VISIBLE_UNIT, HIDDEN_UNIT))
rbm = RBM(VISIBLE_UNIT, HIDDEN_UNIT, CD_K, weights, visible_bias, hidden_bias, use_cuda=CUDA)

train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=BACTH_SIZE)

precision_list, recall_list, f_measure_list, ndcg_list, hit_num_list = [], [], [], [], []

for epoch in range(EPOCH):
    epoch_error = 0.0
    for _, batch in enumerate(train_data_loader):
        if CUDA:
            batch = batch.cuda()
        batch_error = rbm.contrastive_divergence(batch)
        epoch_error += batch_error
    print('Epoch %d' % epoch)
    print('\t Epoch Error: %.4f' % epoch_error)

    pre, rec, f_mea, ndcg, hit_num = evaluate_global_model(rbm.weights, rbm.visible_bias, rbm.hidden_bias,
                                                           user_num, item_num, train_data_loader, test_set,
                                                           train_set_item, CUDA, BACTH_SIZE)

    precision_list.append(pre)
    recall_list.append(rec)
    f_measure_list.append(f_mea)
    ndcg_list.append(ndcg)
    hit_num_list.append(hit_num)


save_as_pkl(RESULT_FOLDER, 'precision', precision_list)
save_as_pkl(RESULT_FOLDER, 'recall', recall_list)
save_as_pkl(RESULT_FOLDER, 'f_measure', f_measure_list)
save_as_pkl(RESULT_FOLDER, 'NDCG', ndcg_list)
save_as_pkl(RESULT_FOLDER, 'hit_num', hit_num_list)

# 文件中读取结果

# precision_list = read_from_pkl(RESULT_FOLDER, 'precision.pkl')
# recall_list = read_from_pkl(RESULT_FOLDER, 'recall.pkl')
# f_measure_list = read_from_pkl(RESULT_FOLDER, 'f_measure.pkl')
# ndcg_list = read_from_pkl(RESULT_FOLDER, 'NDCG.pkl')
# hit_num_list = read_from_pkl(RESULT_FOLDER, 'hit_num.pkl')

print('the statistical information in each federated learning:')
print('\t the best performance of top@[5, 10, 15, 20]')
print('\t \t \t \t round \t \t metrics value')
print('\t \t precision \t %s ---> %s' % (np.argmax(precision_list, axis=0), max(precision_list)))
print('\t \t recall \t %s ---> %s' % (np.argmax(recall_list, axis=0), max(recall_list)))
print('\t \t f_measure \t %s ---> %s' % (np.argmax(f_measure_list, axis=0), max(f_measure_list)))
print('\t \t NDCG \t \t %s ---> %s' % (np.argmax(ndcg_list, axis=0), max(ndcg_list)))
print('\t \t hit_num \t %s ---> %s' % (np.argmax(hit_num_list, axis=0), max(hit_num_list)))
