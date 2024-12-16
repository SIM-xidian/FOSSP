import os

from models.model_cifar import getSSL
from dataset.dataset_cifar import DataSet

#############
# training parameters
feature_dim = 256
version = 'RELEASE_8&9_cifar_pretrain'
epoch_num = 250
lr = 1e-3
batchsize = 256
#############

#############
# set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#############

#############
# unknown_class = set([])
unknown_class = {8, 9}
# unknown_class = {1, 2, 4, 5, 6}
dataset = DataSet(unknown_class)
mods = dataset.get_meta()
X_train, Y_train, X_val, Y_val, X_test, Y_test, train_class = dataset.get_train_val_test()
num_class = len(train_class)
train_mods = [mods[i] for i in train_class]
input_shape = [32, 32]
#############

#############
model_path = './models/model_{}.pkl'.format(version)
model = getSSL(input_shape, feature_dim)
#############
