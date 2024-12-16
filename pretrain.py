import random

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as Data
import sys

from matplotlib import pyplot as plt

import mltools
from centerLoss import nt_xent_loss
from config_pretrain import *

device = torch.device("cuda")

train_dataset = Data.TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
train_loader = Data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=2)
val_dataset = Data.TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val))
val_loader = Data.DataLoader(val_dataset, batch_size=batchsize, shuffle=True, num_workers=2)

# ssl loss
# criterion_sscl = ContrastiveLoss(batch_size=batchsize)
# model optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)


def to_device(device):
    model.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def preprocess(X, way=1, mu=0, sigma=0.01, alpha=np.pi):
    X_preproc = []
    for sample in X:
        if way == 1:
            # 1. Flip
            sample_flip = np.copy(sample)
            sample_flip[0, :] = -sample[0, :]
            sample_flip[1, :] = -sample[1, :]
            X_preproc.append(sample_flip)
        elif way == 2:
            # 2. Gauss Noisy
            sample_gauss = np.copy(sample)
            sample_gauss[0, :] += random.gauss(mu, sigma)
            sample_gauss[1, :] += random.gauss(mu, sigma)
            X_preproc.append(sample_gauss)
        elif way == 3:
            # 3. Rotate
            sample_rotate = np.copy(sample)
            sample_rotate[0, :] = sample[1, :] * np.sin(alpha) + sample[0, :] * np.cos(alpha)
            sample_rotate[1, :] = sample[1, :] * np.cos(alpha) - sample[0, :] * np.sin(alpha)
            X_preproc.append(sample_rotate)
        elif way == 4:
            # 4. random mask
            sample_random_mask = np.copy(sample)
            sample_random_mask[:, random.randint(0, sample_random_mask.shape[1] - 1)] = 0
            X_preproc.append(sample_random_mask)
        elif way == 5:
            # 5. normalize amplitude
            sample_change_amplitude = np.copy(sample)
            sample_change_amplitude[0, :] = sample_change_amplitude[0, :] / np.max(sample_change_amplitude[0, :])
            sample_change_amplitude[1, :] = sample_change_amplitude[1, :] / np.max(sample_change_amplitude[1, :])
            X_preproc.append(sample_change_amplitude)

    return np.array(X_preproc)


def train_torch(start_epoch=1):
    print('Start Training')
    print('Using feature dimension {}, version {}'.format(feature_dim, version))
    print(model)

    if not os.path.isdir('models/' + version):
        os.mkdir('models/' + version)

    if not os.path.isdir('models/' + version + '/tsne/'):
        os.mkdir('models/' + version + '/tsne/')

    model_batch_path = './models/{}/model_{}d_{}.pkl'
    loss_txt_path = f'./models/{version}/loss.txt'

    loss_list_train = []

    loss_list_val = []

    to_device(device)

    model.train()
    with open(loss_txt_path, 'w') as f:
        resultlines = []
        for epoch in range(start_epoch, start_epoch + epoch_num):
            model.train()
            feature_output_train = []
            feature_label_train = []
            for i, data in enumerate(train_loader):
                inputs, labels = data
                aug1, aug2 = preprocess(inputs, 1), preprocess(inputs, 2)
                inputs, labels = inputs.to(device), labels.to(device)
                aug1, aug2 = torch.tensor(aug1).to(device), torch.tensor(aug2).to(device)
                optimizer.zero_grad()

                labels = torch.max(labels.long(), 1)[1]

                feature_output_train.append(model.getSemantic(inputs).cpu().detach().numpy())
                feature_label_train.append(labels.cpu().detach().numpy())

                # loss_sscl_train = criterion_sscl(model(aug1), model(aug2))
                loss_sscl_train = nt_xent_loss(model(aug1), model(aug2))
                loss_train = loss_sscl_train
                loss_train.backward()
                optimizer.step()

            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    inputs, labels = data
                    aug1, aug2 = preprocess(inputs, 1), preprocess(inputs, 2)
                    aug1, aug2 = torch.tensor(aug1).to(device), torch.tensor(aug2).to(device)

                    # loss_sscl_val = criterion_sscl(model(aug1), model(aug2))
                    loss_sscl_val = nt_xent_loss(model(aug1), model(aug2))
                    loss_val = loss_sscl_val

            print('[%d/%d] train loss: %.3f, sscl loss: %.3f\n         val loss: %.3f, sscl loss: %.3f' % (
                epoch, start_epoch + epoch_num - 1, loss_train.item(), loss_sscl_train.item(), loss_val.item(),
                loss_sscl_val.item()))
            resultlines.append(
                '[%d/%d] train loss: %.3f, sscl loss: %.3f\n         val loss: %.3f, sscl loss: %.3f' % (
                    epoch, start_epoch + epoch_num - 1, loss_train.item(), loss_sscl_train.item(), loss_val.item(),
                    loss_sscl_val.item()))

            loss_list_train.append(np.round(loss_train.item(), 3))
            loss_list_val.append(np.round(loss_val.item(), 3))

            if epoch >= 50 and epoch % 25 == 0:
                state = {
                    'backbone': model.backbone.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': start_epoch + epoch
                }
                torch.save(state, model_batch_path.format(version, feature_dim, epoch))
                mltools.tsne(feature_output_train, feature_label_train, train_mods,
                             f'./models/{version}/tsne/tsne_{epoch}.jpg', dataset='cifar')

        f.writelines(resultlines)
    state = {
        'backbone': model.backbone.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': start_epoch + epoch_num
    }
    torch.save(state, model_path)
    print('Finished Training')

    plt.figure()

    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('epochs')  # x轴标签
    plt.ylabel('loss')  # y轴标签

    plt.plot(range(1, epoch_num + 1), loss_list_train, color='blue', linewidth=1, linestyle="solid", label="train loss")
    plt.plot(range(1, epoch_num + 1), loss_list_val, color='red', linewidth=1, linestyle="solid", label="val loss")
    plt.legend(loc=0)
    plt.grid()
    plt.title('Loss curve')
    plt.savefig(f'./models/{version}/loss.jpg')
    # plt.show()
    plt.close()
    np.save(f'./models/{version}/loss_train.npy', loss_list_train)
    np.save(f'./models/{version}/loss_val.npy', loss_list_val)


if __name__ == '__main__':
    start_epoch = 1
    if len(sys.argv) > 1 and sys.argv[1] == '-r':
        print('Resuming model from {}'.format(model_path))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

        print('Resume training from epoch {}'.format(start_epoch))

    train_torch(start_epoch)
