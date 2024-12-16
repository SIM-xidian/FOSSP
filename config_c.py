import os

from models.model import getSR2CNN
from dataset.dataset import DataSet

#############
# training parameters
lam_center = 0.03
# lam_center = 0
lam_encoder = 10
# lam_encoder = 0
feature_dim = 256
version = 'RELEASE_0&3_201604C'
epoch_num = 250
lr = 1e-3
batchsize = 256
#############

#############
# set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#############

#############
# dataset
# __dataset_path = 'dataset/RML2016.10a_dict.pkl'
__dataset_path = 'dataset/2016.04C.multisnr.pkl'
# unknown_class = set([])
unknown_class = {0, 3}
dataset = DataSet(__dataset_path, unknown_class)
X, lbl, snrs, mods = dataset.get_meta()
X_train, Y_train, X_val, Y_val, X_test, Y_test, train_class = dataset.get_train_val_test()
num_class = len(train_class)
train_mods = [mods[i] for i in train_class]
#############

#############
pretrain_path = './models/model_{}_pretrain.pkl'.format(version)
model_path = './models/model_{}.pkl'.format(version)
model = getSR2CNN(num_class, feature_dim)
#############
