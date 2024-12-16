import os

from resnet.model import resnet34, resnet50, resnet101
from dataset.dataset import DataSet

#############
# training parameters
lam_center = 0
# lam_center = 0
lam_encoder = 0
# lam_encoder = 0
feature_dim = 256
version = 'RELEASE_0&2&3&4&5&6&9_resnet34_200e'
epoch_num = 200
lr = 1e-3
batchsize = 256
#############

#############
# set GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#############

#############
# dataset
__dataset_path = 'dataset/RML2016.10a_dict.pkl'
# unknown_class = set([])
# unknown_class = {2, 5}
unknown_class = {0, 2, 3, 4, 5, 6, 9}
dataset = DataSet(__dataset_path, unknown_class)
X, lbl, snrs, mods = dataset.get_meta()
X_train, Y_train, X_val, Y_val, X_test, Y_test, train_class = dataset.get_train_val_test()
num_class = len(train_class)
train_mods = [mods[i] for i in train_class]
#############

#############
model_path = './models/model_{}.pkl'.format(version)
model = resnet34(num_classes=num_class)
#############
