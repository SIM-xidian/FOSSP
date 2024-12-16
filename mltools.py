"Adapted from the code (https://github.com/leena201818/radiom) contributed by leena201818"
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_roc(y, y_pred_scores, name, color, save_filename):
    plt.figure(save_filename)
    fpr, tpr, thresholds = roc_curve(y, y_pred_scores, pos_label=1)
    plt.plot(fpr, tpr, lw=5, label='{} (AUC={:.3f})'.format(name, auc(fpr, tpr)), color=color)
    plt.plot([0, 1], [0, 1], '--', lw=5, color='grey')
    plt.axis('square')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(save_filename)
    plt.close()
    return fpr, tpr


def calculate_confusion_matrix(Y, Y_hat, train_class, mods, label):
    n_classes = len(label)
    conf = np.zeros([n_classes, n_classes])
    confnorm = np.zeros([n_classes, n_classes])

    for i in range(Y.shape[0]):
        m = list(Y[i, :]).index(1)
        n = list(Y_hat[i, :]).index(1)
        conf[m, n] = conf[m, n] + 1

    for i in range(len(mods) - len(train_class)):
        index = np.argmax(conf[len(train_class) + i, :])
        t = conf[len(train_class) + i, index]
        conf[len(train_class) + i, index] = conf[len(train_class) + i, len(train_class) + i]
        conf[len(train_class) + i, len(train_class) + i] = t

    confnorm[0:len(mods), :] = conf[0:len(mods), :].astype('float') / conf[0:len(mods), :].sum(axis=1)[:, np.newaxis]

    return confnorm


def calculate_confusion_matrix1(Y, Y_hat, label):
    n_classes = len(label)
    conf = np.zeros([n_classes, n_classes])
    confnorm = np.zeros([n_classes, n_classes])

    for i in range(Y_hat.shape[0]):
        m = Y[i]
        n = Y_hat[i]
        conf[m, n] += 1

    confnorm[:, :] = conf[:, :].astype('float') / conf[:, :].sum(axis=1)[:, np.newaxis]

    return confnorm


def plot_confusion_matrix(cm, xlabels=None, ylabels=None, train_class=None, untrain_class=None, save_filename=None):
    cm = np.around(cm, decimals=2)
    if xlabels is None:
        xlabels = []
    if ylabels is None:
        ylabels = []
    fig, ax = plt.subplots(figsize=(4, 3), dpi=600)
    plt.imshow(cm * 100, interpolation='nearest', cmap=plt.get_cmap("Blues"))
    plt.colorbar()
    xtick_marks = np.arange(len(xlabels))
    ytick_marks = np.arange(len(ylabels))
    plt.xticks(xtick_marks, xlabels, rotation=90, size=6)
    plt.yticks(ytick_marks, ylabels, size=6)

    for i in range(len(untrain_class)):
        ax.get_xticklabels()[i + len(train_class)].set_color("red")
        ax.get_yticklabels()[i + len(train_class)].set_color("red")

    for j in range(len(ytick_marks)):
        for i in range(len(ytick_marks)):
            if i != j:
                if int(np.around(cm[i, j] * 100)) == 100:
                    plt.text(j, i, int(np.around(cm[i, j] * 100)), ha="center", va="center", fontsize=7)
                else:
                    plt.text(j, i, int(np.around(cm[i, j] * 100)), ha="center", va="center", fontsize=10)
            elif i == j:
                if int(np.around(cm[i, j] * 100)) == 100:
                    plt.text(j, i, int(np.around(cm[i, j] * 100)), ha="center", va="center", fontsize=7,
                                    color='darkorange')
                else:
                    plt.text(j, i, int(np.around(cm[i, j] * 100)), ha="center", va="center", fontsize=10,
                                    color='darkorange')
        for i in range(len(ytick_marks), len(xtick_marks)):
            if int(np.around(cm[i, j] * 100)) == 100:
                plt.text(j, i, int(np.around(cm[i, j] * 100)), ha="center", va="center", fontsize=7)
            else:
                plt.text(j, i, int(np.around(cm[i, j] * 100)), ha="center", va="center", fontsize=10)

    for j in range(len(ytick_marks), len(xtick_marks)):
        for i in range(len(xtick_marks)):
            if int(np.around(cm[i, j] * 100)) == 100:
                plt.text(j, i, int(np.around(cm[i, j] * 100)), ha="center", va="center", fontsize=7)
            else:
                plt.text(j, i, int(np.around(cm[i, j] * 100)), ha="center", va="center", fontsize=10)

    plt.tight_layout()
    if save_filename is not None:
        plt.savefig(save_filename, dpi=600, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix1(cm, xlabels=None, ylabels=None, train_class=None, untrain_class=None, save_filename=None):
    cm = np.around(cm, decimals=2)
    if xlabels is None:
        xlabels = []
    if ylabels is None:
        ylabels = []
    fig, ax = plt.subplots(figsize=(4, 3), dpi=600)
    plt.imshow(cm * 100, interpolation='nearest', cmap=plt.get_cmap("Blues"))
    plt.colorbar()
    xtick_marks = np.arange(len(xlabels))
    ytick_marks = np.arange(len(ylabels))
    plt.xticks(xtick_marks, xlabels, rotation=90, size=6)
    plt.yticks(ytick_marks, ylabels, size=6)

    for i in range(len(untrain_class)):
        ax.get_yticklabels()[i + len(train_class)].set_color("red")

    for j in range(len(ytick_marks)):
        for i in range(len(xtick_marks)):
            if i != j:
                if int(np.around(cm[i, j] * 100)) == 100:
                    plt.text(j, i, int(np.around(cm[i, j] * 100)), ha="center", va="center", fontsize=7)
                else:
                    plt.text(j, i, int(np.around(cm[i, j] * 100)), ha="center", va="center", fontsize=10)
            elif i == j:
                if int(np.around(cm[i, j] * 100)) == 100:
                    plt.text(j, i, int(np.around(cm[i, j] * 100)), ha="center", va="center", fontsize=7,
                                    color='darkorange')
                else:
                    plt.text(j, i, int(np.around(cm[i, j] * 100)), ha="center", va="center", fontsize=10,
                                    color='darkorange')
        for i in range(len(xtick_marks), len(ytick_marks)):
            if int(np.around(cm[i, j] * 100)) == 100:
                plt.text(j, i, int(np.around(cm[i, j] * 100)), ha="center", va="center", fontsize=7)
            else:
                plt.text(j, i, int(np.around(cm[i, j] * 100)), ha="center", va="center", fontsize=10)
    plt.tight_layout()
    if save_filename is not None:
        plt.savefig(save_filename, dpi=600, bbox_inches='tight')
    plt.close()


def tsne(features_output1, label, mods, save_filename, dataset='', num_known_class=None):
    colors = ['lawngreen', 'blue', 'darkorange', 'yellow', 'springgreen', 'purple', 'red', 'greenyellow', 'slateblue',
              'deepskyblue',
              'forestgreen', 'black']
    # -----------------load model-----------------
    model = TSNE(n_components=2, random_state=2022, learning_rate='auto', init='pca')
    # -----------------load model-----------------

    features_output1 = np.vstack(features_output1)
    label = np.hstack(label)

    features_output2 = model.fit_transform(features_output1)
    x_max = np.max(features_output2[:, 0]) + 5
    x_min = np.min(features_output2[:, 0]) - 5
    y_max = np.max(features_output2[:, 1]) + 5
    y_min = np.min(features_output2[:, 1]) - 5
    if dataset == 'cifar':
        num1 = 1.235
    else:
        num1 = 1.2
    num2 = 1.0
    num3 = 0
    num4 = 0
    plt.figure()
    for i in np.unique(label):
        i = int(i)
        idx = np.where(label == i)[0]
        plt.scatter(features_output2[idx, 0], features_output2[idx, 1], c=colors[i], label=mods[i])
    if num_known_class is None:
        plt.legend(fontsize=8, bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)
    else:
        labelcolor = ['black' for i in range(num_known_class)] + ['red' for i in range(len(mods) - num_known_class)]
        plt.legend(fontsize=8, bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4, labelcolor=labelcolor)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(right=0.83)
    plt.savefig(save_filename)
    # plt.show()
    plt.close()


# def plot_snr(val_snr_labels, val_true_labels, val_pred_labels, modulation_classes, data_set, save_filename):
#     train_oa, val_oa = [], []
#     val_true_labels, val_pred_labels = np.array(val_true_labels), np.array(val_pred_labels)
#     for j in np.unique(val_true_labels):
#         idx = np.where(val_true_labels == j)[0]
#         oa = []
#         for i in np.unique(val_snr_labels):  # 计算不同信噪比的准确率
#             idx1 = np.where(val_snr_labels[idx] == i)[0]
#             oa.append(accuracy_score(val_true_labels[idx][idx1], val_pred_labels[idx][idx1]))
#         val_oa.append(oa)
#
#     # 不同信噪比下准确率的折线图
#     x = range(np.unique(val_snr_labels).shape[0])
#     k = 0
#     marker = ['o', 'v', 's', 'x', 'p', '+']
#     for i in np.unique(val_true_labels):
#         kk = int(k / len(marker))
#         plt.plot(x, val_oa[k], marker=marker[kk], label=f'{modulation_classes[i]}')
#         k += 1
#     plt.ylim(0, 1)
#     plt.legend(bbox_to_anchor=(1.0, 1.1))  # 让图例生效
#     plt.xticks(x, np.unique(val_snr_labels), rotation=45)
#     plt.margins(0)
#     plt.subplots_adjust(bottom=0.15)
#     plt.xlabel("SNRS")  # X轴标签
#     plt.ylabel("Accuracy")  # Y轴标签
#
#     plt.tight_layout()
#     plt.savefig(save_filename)
#     plt.close()


if __name__ == '__main__':
    print()
