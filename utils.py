import numpy as np
import os
import pickle
import torch

def save_as_csv(data_dir, file_name, data):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    np.savetxt(data_dir+file_name, data, delimiter=',')

def save_as_npy(data_dir, file_name, data):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    np.save(data_dir+file_name, data)

def save_as_pkl(data_dir, filename, data):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    filehandler = open(data_dir + "/" + filename + ".pkl", "wb")
    pickle.dump(data, filehandler)
    filehandler.close()

def save_as_pt(data_dir, filename, data):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    torch.save(data, data_dir + filename)

def read_from_npy(data_dir, file_name):
    return np.load(data_dir + file_name).item()

def read_from_pkl(save_dir, filename):
    return pickle.load(open(save_dir + filename , 'rb'))

def read_from_pt(save_dir, filename):
    return torch.load(save_dir + filename)


class Params:
    def __init__(self, b, c, client_num, CUDA, e, max_communication_round, rbm_visible_unit, rbm_hidden_unit, rbm_k):
        self.b = b
        self.c = c
        self.client_num = client_num
        self.CUDA = CUDA
        self.e = e
        self.max_communication_round = max_communication_round

        # RBM parameters
        self.rbm_visible_unit = rbm_visible_unit
        self.rbm_hidden_unit = rbm_hidden_unit
        self.rbm_k = rbm_k


