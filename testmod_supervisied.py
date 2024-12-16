import numpy as np
import torch
import torch.utils.data as Data
import sys

import mltools
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import *

known_class = set(range(len(mods))) - unknown_class
class_sort = list(known_class) + list(unknown_class)


def RESULT_LOGGER(result_list, message):
    result_list.append('{}\n'.format(message))
    print(message)


def cal_acc_evol(test_loader, result_path):
    with torch.no_grad():
        model.eval()
        feature_output_test, feature_label_test = [], []
        pred_labels_test, true_labels_test = [], []
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            labels = torch.max(labels.long(), 1)[1]

            feature_output_test.append(model.getSemantic(inputs).cpu().detach().numpy())
            feature_label_test.append(labels.cpu().detach().numpy())

            pred_labels_test.append(np.argmax(outputs.cpu().detach().numpy(), axis=1))
            true_labels_test.append(labels.cpu().detach().numpy())
    true_labels_test = np.hstack(true_labels_test)
    pred_labels_test = np.hstack(pred_labels_test)
    conf = np.zeros([len(mods), len(mods)])
    for i in range(pred_labels_test.shape[0]):
        m = true_labels_test[i]
        n = pred_labels_test[i]
        conf[m, n] += 1
    mltools.tsne(feature_output_test, feature_label_test, train_mods, f'./results/{version}/tsne/tsne.jpg')
    confnorm = mltools.calculate_confusion_matrix1(true_labels_test, pred_labels_test, mods)
    mltools.plot_confusion_matrix(
        confnorm,
        xlabels=mods,
        ylabels=mods,
        train_class=train_class,
        untrain_class=[],
        save_filename=f'./results/{version}/confusion_matrix/ConfusionMatrix.png'
    )

    # 计算 overall accuracy
    oa = np.trace(conf) / conf.sum()
    # # 计算各类别 precision和 recall
    # precision_cls = np.diag(conf) / conf.sum(axis=1)
    # recall_cls = np.diag(conf) / conf.sum(axis=0)
    # # 计算各类别 f1-score
    # f1_cls = (2 * precision_cls * recall_cls) / (precision_cls + recall_cls)
    # # 计算 mean f1-score
    # mf1 = np.nanmean(f1_cls)

    aa = 0
    f1 = f1_score(true_labels_test, pred_labels_test, average='weighted')

    resultlines = []
    with open(result_path, 'w') as f:

        for i in class_sort:
            aa += accuracy_score(true_labels_test[true_labels_test == i], pred_labels_test[true_labels_test == i])
            RESULT_LOGGER(resultlines, "accuracy(class:{}): {}".format(i, accuracy_score(
                true_labels_test[true_labels_test == i],
                pred_labels_test[true_labels_test == i]
            )))

        RESULT_LOGGER(resultlines, "\nAA: {}".format(aa / len(class_sort)))
        RESULT_LOGGER(resultlines, "OA: {}".format(oa))
        RESULT_LOGGER(resultlines, "mean F1-score: {}".format(f1))
        f.writelines(resultlines)


if __name__ == '__main__':
    if torch.cuda.is_available():
        print('GPU is available...')

    model_paths = []
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isdir(path):
            model_paths = [os.path.join(path, x) for x in os.listdir(path)]
            version = list(filter(lambda x: not x == '', path.split('/')))[-1]
        elif os.path.isfile(path):
            model_paths.append(path)
    else:
        model_paths.append(model_path)

    if not os.path.isdir('results/' + version):
        os.mkdir('results/' + version)

    if not os.path.isdir('results/' + version + '/confusion_matrix/'):
        os.mkdir('results/' + version + '/confusion_matrix/')

    if not os.path.isdir('results/' + version + '/tsne/'):
        os.mkdir('results/' + version + '/tsne/')

    result_path = 'results/' + version + '/result.txt'
    device = torch.device("cuda")
    test_dataset = Data.TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_loader = Data.DataLoader(test_dataset, batch_size=batchsize, shuffle=True, num_workers=2)
    model.to(device)

    for model_path in model_paths:
        print('Loading model from {}'.format(model_path))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        zero_shot_path = 'results/' + version + '/ZSL_' + os.path.split(model_path)[-1][:-4] + '.txt'
        print('With {} epochs training...'.format(checkpoint['epoch'] - 1))
        cal_acc_evol(test_loader, result_path)
    print('test end')
