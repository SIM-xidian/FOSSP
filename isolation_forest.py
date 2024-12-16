import torch
import numpy as np
from sklearn.ensemble import IsolationForest
from torch.utils.data import TensorDataset, DataLoader
from config_resnet import *
import mltools


def RESULT_LOGGER(result_list, message):
    result_list.append('{}\n'.format(message))
    print(message)


if __name__ == '__main__':

    if not os.path.isdir('results/' + version + '_isolation_forest'):
        os.mkdir('results/' + version + '_isolation_forest')

    if not os.path.isdir('results/' + version + '_isolation_forest/confusion_matrix/'):
        os.mkdir('results/' + version + '_isolation_forest/confusion_matrix/')

    if not os.path.isdir('results/' + version + '_isolation_forest/roc/'):
        os.mkdir('results/' + version + '_isolation_forest/roc/')

    device = torch.device("cuda")
    model = model.to(device)
    model.load_state_dict(torch.load(model_path)['model'])
    model.eval()

    dataset.split_unknown()
    train_map, test_map, unknown_test_map = dataset.get_train_test_maps()

    train_samples = np.concatenate((*(train_map.values()),), 0)
    train_labels = np.concatenate(
        (*map(lambda x: np.full([x[1].shape[0]], x[0], dtype=np.int64), train_map.items()),), 0)

    tackled_test_data = np.concatenate((*(test_map.values()),), 0)
    tackled_label = np.concatenate(
        (*map(lambda x: np.full([x[1].shape[0]], x[0], dtype=np.int64), test_map.items()),), 0)
    tackled_unknown_test_data = np.concatenate((*(unknown_test_map.values()),), 0)
    tackled_unknown_label = np.concatenate(
        (*map(lambda x: np.full([x[1].shape[0]], x[0], dtype=np.int64), unknown_test_map.items()),), 0)
    test_samples = np.concatenate((tackled_test_data, tackled_unknown_test_data), 0)
    test_labels = np.concatenate((tackled_label, tackled_unknown_label), 0)

    train_dataset = TensorDataset(torch.from_numpy(train_samples), torch.from_numpy(train_labels))
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=True)

    val_dataset = TensorDataset(torch.from_numpy(test_samples), torch.from_numpy(test_labels))
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=int(len(test_samples)/5), shuffle=False)

    outliers_fraction = 0.3
    algorithm = IsolationForest(contamination=outliers_fraction, random_state=42)

    scores, pred_labels, labels = [], [], []
    model.eval()
    with torch.no_grad():
        # for batch_inputs, _ in train_dataloader:
        #     batch_inputs = batch_inputs.float().cuda()
        #     X = model.getSemantic(batch_inputs).detach().cpu().numpy()
        #     algorithm.fit(X)

        for batch_inputs, batch_labels in val_dataloader:
            batch_inputs = batch_inputs.float().cuda()
            batch_labels = batch_labels.long().cuda()
            X = model.getSemantic(batch_inputs).detach().cpu().numpy()
            algorithm.fit(X)
            pred_labels.append(algorithm.predict(X))
            scores.append(algorithm.decision_function(X))
            labels.append(batch_labels)

    pred_labels = np.concatenate(pred_labels, axis=0)
    scores = np.concatenate(scores, axis=0)
    labels = torch.cat(labels, dim=0).detach().cpu().numpy()
    pred_labels = np.array(pred_labels)
    labels = np.array(labels)
    scores = np.array(scores)

    untrain_class = []
    for i in range(len(mods)):
        if i not in train_class:
            untrain_class.append(i)
    new_xlabels = ['known', 'unknown']
    new_ylabels = [
        ','.join([mods[i] for i in train_class]),
        ','.join([mods[i] for i in untrain_class]),
    ]

    pred_isolation_forest = np.array(pred_labels)
    pred_isolation_forest[pred_isolation_forest == 1] = 0
    pred_isolation_forest[pred_isolation_forest == -1] = 1

    labels[labels < num_class] = 0
    labels[labels >= num_class] = 1

    confnorm = mltools.calculate_confusion_matrix1(labels, pred_isolation_forest, new_xlabels)
    mltools.plot_confusion_matrix1(confnorm,
                                   xlabels=new_xlabels,
                                   ylabels=new_xlabels,
                                   train_class=['known'],
                                   untrain_class=['unknown'],
                                   save_filename=f'results/{version}_isolation_forest/confusion_matrix/ConfusionMatrix.png')

    y, y_pred = labels.copy(), pred_isolation_forest.copy()
    y[y == 1] = -1
    y[y == 0] = 1
    # y_pred[y_pred == 1] = -1
    # y_pred[y_pred == 0] = 1
    fpr, tpr = mltools.plot_roc(
        y,
        scores,
        name='Isolation Forest',
        color='green',
        save_filename=f'results/{version}_isolation_forest/roc/ROC.png'
    )

    np.save(f'results/{version}_isolation_forest/fpr.npy', fpr)
    np.save(f'results/{version}_isolation_forest/tpr.npy', tpr)

    zero_shot_path = 'results/' + version + '_isolation_forest/' + os.path.split(model_path)[-1][:-4] + '.txt'
    with open(zero_shot_path, 'w') as f:
        resultlines = []
        oa = np.trace(confnorm) / confnorm.sum()
        precision_cls = np.diag(confnorm) / confnorm.sum(axis=1)
        recall_cls = np.diag(confnorm) / confnorm.sum(axis=0)
        f1_cls = (2 * precision_cls * recall_cls) / (precision_cls + recall_cls)
        f1 = np.nanmean(f1_cls)
        RESULT_LOGGER(resultlines, "OA: {}".format(oa))
        RESULT_LOGGER(resultlines, f'mean F1-score: {f1}')
        f.writelines(resultlines)

