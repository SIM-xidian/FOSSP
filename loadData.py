import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_data():
    with open(r"dataset/RML2016.10b.dat", 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        # data.keys() = ['8psk', -10]
        # print(len(data.keys()))
        # print(data.keys())
        num = 0
        for key in data.keys():
            if key[1] >= 16:
                # print(key[1])
                num += 1
        print(num)
        snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], data.keys())))), [1, 0])
        X = []
        lbl = []
        for mod in mods:
            for snr in snrs:
                X.append(data[(mod, snr)])
                for i in range(data[(mod, snr)].shape[0]): lbl.append((mod, snr))
        X = np.vstack(X)

        np.random.seed(2022)
        n_examples = X.shape[0]
        n_train = n_examples * 0.8  # 划分数据集 n_train : n_test = 8:2
        train_idx = np.random.choice(range(0, int(n_examples)), size=int(n_train), replace=False)
        test_idx = list(set(range(0, n_examples)) - set(train_idx))
        X_train = X[train_idx]
        X_test = X[test_idx]

        def to_onehot(yy):
            yy1 = np.zeros([len(yy), max(yy) + 1])
            yy1[np.arange(len(yy)), yy] = 1
            return yy1

        Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
        Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))

        X_train = torch.from_numpy(X_train)
        X_train = X_train.unsqueeze(1)  # [176000, 1, 2, 128]
        Y_train = torch.from_numpy(Y_train)
        X_test = torch.from_numpy(X_test)
        X_test = X_test.unsqueeze(1)  # [44000, 1, 2, 128]
        Y_test = torch.from_numpy(Y_test)

        return X_train, Y_train, X_test, Y_test, mods


def plot_wave(data, label, mods):
    print(data.shape)  # [17600, 1, 2, 128]
    print(len(label))  # (17600,)
    mod_class = 1
    for i in range(25):
        if label[i] == mod_class:
            sample = data[i, 0, :, :]

            # plt.figure(figsize=(20, 10))
            # plt.subplot(211)
            # plt.plot(sample[0, :], color='red')
            # plt.subplot(212)
            # plt.plot(sample[1, :], color='blue')
            # plt.savefig(f'./images/{i}-{mods[mod_class]}-single.jpg')
            # plt.tight_layout()
            # plt.close()

            sample1 = np.copy(sample)
            plt.figure(figsize=(20, 10))
            plt.plot(sample1[0, :], color='red')
            plt.plot(sample1[1, :], color='blue')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.title(f'{mods[mod_class]}')
            plt.savefig(f'./images/{i}-{mods[mod_class]}-double.jpg')
            plt.tight_layout()
            plt.close()

            sample2 = np.copy(sample)
            sample2[0, :] = -sample[0, :]
            sample2[1, :] = -sample[1, :]
            plt.figure(figsize=(20, 10))
            plt.plot(sample2[0, :], color='red')
            plt.plot(sample2[1, :], color='blue')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.title(f'{mods[mod_class]}')
            plt.savefig(f'./images/{i}-{mods[mod_class]}-double_flip.jpg')
            plt.tight_layout()
            plt.close()

            sample3 = np.copy(sample)
            sample3[0, :] += random.gauss(0, 0.01)
            sample3[1, :] += random.gauss(0, 0.01)
            plt.figure(figsize=(20, 10))
            plt.plot(sample3[0, :], color='red')
            plt.plot(sample3[1, :], color='blue')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.title(f'{mods[mod_class]}')
            plt.savefig(f'./images/{i}-{mods[mod_class]}-double_gauss.jpg')
            plt.tight_layout()
            plt.close()

            sample4 = np.copy(sample)
            alpha = np.pi
            sample4[0, :] = sample[1, :] * np.sin(alpha) + sample[0, :] * np.cos(alpha)
            sample4[1, :] = sample[1, :] * np.cos(alpha) - sample[0, :] * np.sin(alpha)
            plt.figure(figsize=(20, 10))
            plt.plot(sample4[0, :], color='red')
            plt.plot(sample4[1, :], color='blue')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.title(f'{mods[mod_class]}')
            plt.savefig(f'./images/{i}-{mods[mod_class]}-double_rotate.jpg')
            plt.tight_layout()
            plt.close()

            # fre, t, zxx = signal.stft(sample[0, :], nperseg=128)
            # plt.pcolormesh(t, fre, np.abs(zxx), shading='auto')
            # plt.axis('off')
            # plt.savefig(f'./images/{i}-{mods[mod_class]}-stft.jpg')
            # plt.tight_layout()
            # plt.close()


def plot_allMods(data, mods):
    allMods = [11, 24, 3, 19, 21, 2, 6, 31, 7, 13, 1]

    plt.figure(figsize=(20, 10))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    for i in range(len(allMods)):
        sample = data[allMods[i], 0, :, :]
        ax = plt.subplot2grid((3, 4), (int(i / 4), int(i % 4)))
        ax.plot(sample[0, :], color='red')
        ax.plot(sample[1, :], color='blue')
        ax.set_title(f'{mods[i]}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')

    plt.savefig(f'./images/allMods_2016.04c.jpg')
    plt.tight_layout()
    plt.close()


if __name__ == '__main__':

    # load data
    X_train, Y_train, X_test, Y_test, mods = load_data()

    # remove onehot
    label_train = [one_label.tolist().index(1) for one_label in Y_train]
    label_test = [one_label.tolist().index(1) for one_label in Y_test]

    # plot time frequency wave figure
    plot_wave(X_train, label_train, mods)

    # plot_allMods(X_train, mods)

    # data = X_train.reshape(176000, 2 * 128)
    # label = label_train
    #
    # print('Computing t-SNE embedding')
    # tsne = TSNE(n_components=2, init='pca', random_state=0)
    # t0 = time.time()
    # result = tsne.fit_transform(data)
    # fig = plot_embedding(result, label, 't-SNE embedding of the digits')
    # plt.show(fig)

