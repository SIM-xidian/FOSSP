import shutil

import numpy as np
import torch
import sys
import copy

from matplotlib import pyplot as plt
from sklearn.metrics import f1_score

import mltools
from config import *

device = torch.device("cuda")


def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy) + 1])
    yy1[np.arange(len(yy)), yy] = 1
    return yy1


def get_semantic_ndarray(data):
    tensor_device = torch.Tensor(data).to(device)
    return model.getSemantic(tensor_device).cpu().numpy()


def calculate_distance(x, transform_matrix):
    return np.sqrt(np.dot(np.dot(x, transform_matrix), x.transpose()))


def RESULT_LOGGER(result_list, message):
    result_list.append('{}\n'.format(message))
    print(message)


def gen_sematic_vec(train_map):
    semantic_center_map = {}
    cov_inv_map = {}
    cov_inv_diag_map = {}
    sigma_identity_map = {}
    distance_map = {}

    for certain_class, train_data in train_map.items():
        raw_output = get_semantic_ndarray(train_data)
        semantic_center_map[certain_class] = np.mean(raw_output, 0)

        covariance_mat = np.cov(raw_output, rowvar=False, bias=True)
        cov_inv_map = np.linalg.pinv(covariance_mat)

        cov_inv_diag_mat = np.diagflat(1 / (covariance_mat.diagonal()))
        cov_inv_diag_mat[cov_inv_diag_mat == np.inf] = 0.0
        cov_inv_diag_map[certain_class] = cov_inv_diag_mat
        sigma = np.mean(np.diagflat(covariance_mat.diagonal()))
        sigma_identity_map[certain_class] = 1 / sigma * np.eye(covariance_mat.shape[0])

    distance_map['Maha'] = cov_inv_map
    distance_map['MahaDiag'] = cov_inv_diag_map
    distance_map['SigmaEye'] = sigma_identity_map

    return semantic_center_map, distance_map


def classify_evol(transform_map, semantic_center_map, semantic_vector, coef, coef_unknown):
    predicted_label = -1
    min_dist = float('inf')
    min_dist_recorded = float('inf')
    dists_known_I = []
    if_known = False

    for certain_class in range(num_class):
        semantic_center = semantic_center_map[certain_class]
        dist = calculate_distance(semantic_vector - semantic_center, transform_map[certain_class])

        eyeMat = np.eye(semantic_center_map[certain_class].shape[0])
        dist_I = calculate_distance(semantic_vector - semantic_center, eyeMat)

        dists_known_I.append(dist_I)

        if dist < 3 * np.sqrt(semantic_vector.shape[0]) * coef:
            if_known = True

        if dist < min_dist:
            min_dist = dist
            predicted_label = certain_class

    mean_dist = np.mean(dists_known_I)
    min_dist = min(dists_known_I)
    dist_svm = 3 * np.sqrt(semantic_vector.shape[0]) * coef - min_dist

    if not if_known:
        # first unknown instance shows up
        if len(semantic_center_map.keys()) == num_class:
            predicted_label = -1
        else:
            if_recorded = False
            recorded_unknowns = set(semantic_center_map.keys()) - set(list(range(num_class)))
            for recorded_unknown_class in recorded_unknowns:
                semantic_center = semantic_center_map[recorded_unknown_class]
                dist = calculate_distance(semantic_vector - semantic_center, eyeMat)
                if dist <= coef_unknown * (min_dist + mean_dist) / 2:
                    if_recorded = True
                    break

            if if_recorded:
                for recorded_unknown_class in recorded_unknowns:
                    semantic_center = semantic_center_map[recorded_unknown_class]
                    dist = calculate_distance(semantic_vector - semantic_center, eyeMat)
                    if dist < min_dist_recorded:
                        min_dist_recorded = dist
                        predicted_label = recorded_unknown_class
            else:
                predicted_label = -1

    return predicted_label, dist_svm


# include the one shot sample
def cal_acc_evol(train_map, test_map, unknown_test_map, version, distance='MahaDiag'):
    semantic_center_map_origin, distance_map = gen_sematic_vec(train_map)

    print('Using cluster plus mode')
    with open(zero_shot_path, 'w') as f:
        tackled_test_data = np.concatenate((*(test_map.values()),), 0)
        tackled_label = np.concatenate(
            (*map(lambda x: np.full([x[1].shape[0]], x[0], dtype=np.int64), test_map.items()),), 0)
        tackled_unknown_test_data = np.concatenate((*(unknown_test_map.values()),), 0)
        tackled_unknown_label = np.concatenate(
            (*map(lambda x: np.full([x[1].shape[0]], x[0], dtype=np.int64), unknown_test_map.items()),), 0)
        test_samples = np.concatenate((tackled_test_data, tackled_unknown_test_data), 0)
        test_labels = np.concatenate((tackled_label, tackled_unknown_label), 0)
        predicted_semantics = get_semantic_ndarray(test_samples)

        coef_unknown = 1
        coef = 0.05

        coef_list = []
        tkr_list = []
        tur_list = []
        oa_list = []
        f1_list = []

        while coef <= 1.0:
            coef_list.append(coef)
            resultlines = []
            RESULT_LOGGER(resultlines, 'Distance {} with coefficient {}'.format(distance, round(coef, 2)))

            # shuffle test samples
            indices = np.random.permutation(test_samples.shape[0])
            predicted_semantics = predicted_semantics[indices]
            test_labels = test_labels[indices]

            semanticMap = copy.deepcopy(semantic_center_map_origin)

            transform_map = distance_map[distance]

            num_known_total = tackled_test_data.shape[0]  # total number of known samples
            num_known_unknown = 0  # number of known samples discriminated as unknown

            conf = np.zeros([len(train_class), len(train_class)])

            num_unknown_unknown = 0  # number of unknown samples discriminated as unknown
            num_unknown_total = tackled_unknown_test_data.shape[0]  # total number of unknown samples
            new_class_instances_map = {}
            new_class_labels_count_map = {}

            new_class_index = num_class
            predicted_labels = []
            dists_svm = []

            for certain_class, predicted_semantic in zip(test_labels, predicted_semantics):
                predicted_label, dist_svm = classify_evol(transform_map, semanticMap, predicted_semantic, coef, coef_unknown)
                predicted_labels.append(predicted_label)
                dists_svm.append(dist_svm)

                # known class
                if predicted_label in range(num_class):
                    if certain_class in range(num_class):
                        conf[certain_class, predicted_label] += 1
                # unknown class
                else:
                    if certain_class in range(num_class):
                        num_known_unknown += 1
                    else:
                        num_unknown_unknown += 1

                    # new unknown class discriminated
                    if predicted_label == -1:
                        # initialize new semantic center
                        semanticMap[new_class_index] = predicted_semantic
                        new_class_instances_map[new_class_index] = [predicted_semantic]
                        new_class_labels_count_map[new_class_index] = {int(certain_class): 1}
                        new_class_index += 1
                    # classified as newly recorded class
                    else:
                        # update semantic center
                        new_class_instances_map[predicted_label].append(predicted_semantic)
                        semanticMap[predicted_label] = np.mean(new_class_instances_map[predicted_label], axis=0)
                        new_class_labels_count_map[predicted_label][int(certain_class)] = new_class_labels_count_map[
                                                                                                predicted_label].get(
                            int(certain_class), 0) + 1

            for certain_class, test_data in test_map.items():
                RESULT_LOGGER(resultlines, "Accuracy(class:{}):{}".format(train_class[certain_class],
                                                                            conf[certain_class, certain_class] /
                                                                            test_data.shape[0]))
            precision_cls = np.diag(conf) / conf.sum(axis=1)
            recall_cls = np.diag(conf) / conf.sum(axis=0)
            f1_cls = (2 * precision_cls * recall_cls) / (precision_cls + recall_cls)
            mf1 = np.nanmean(f1_cls)
            RESULT_LOGGER(resultlines, 'Known OA: {}'.format(np.trace(conf) / num_known_total))
            RESULT_LOGGER(resultlines, 'Known F1-score: {}'.format(mf1))

            false_unknown = num_known_unknown / num_known_total
            RESULT_LOGGER(resultlines, 'False Unknown Rate: {}\n'.format(false_unknown))

            true_known = (num_known_total - num_known_unknown) / num_known_total
            tkr_list.append(true_known * 100)
            true_unknown = num_unknown_unknown / num_unknown_total
            tur_list.append(true_unknown * 100)

            # FUR is not qualified
            # if false_unknown <= 0.2:

            RESULT_LOGGER(resultlines, 'True known Rate: {}'.format(true_known))
            RESULT_LOGGER(resultlines, 'True Unknown Rate: {}'.format(true_unknown))
            RESULT_LOGGER(resultlines,
                            'Total number of newly identified class: {}'.format(len(new_class_labels_count_map)))

            new_class_candidate_label_map = {}
            for new_class, labels_count_map in new_class_labels_count_map.items():
                class_labels_sorted = sorted(labels_count_map.items(), key=lambda x: x[1], reverse=True)
                new_class_candidate_label_map[new_class] = class_labels_sorted[0]

            for unknown_class_label, test_data in unknown_test_map.items():
                # element: (new_class_label, (candidate_label, count))
                isotopic_new_classes = list(
                    filter(lambda x: x[1][0] == unknown_class_label, new_class_candidate_label_map.items()))
                if len(isotopic_new_classes) < 1:
                    RESULT_LOGGER(resultlines, 'Unknown class {} fails to identify'.format(unknown_class_label))
                    break

                dominant_new_classes = max(isotopic_new_classes, key=lambda x: x[1][1])
                RESULT_LOGGER(resultlines,
                                'Unknown Class {}'.format(list(unknown_class)[unknown_class_label - num_class]))
                RESULT_LOGGER(resultlines, '    Accuracy: {}'.format(dominant_new_classes[1][1] / test_data.shape[0]))
                RESULT_LOGGER(resultlines, '    Precision: {}'.format(
                    dominant_new_classes[1][1] / len(new_class_instances_map[dominant_new_classes[0]]))
                                )

            # TUR is qualified, write results
            # if true_unknown > 0.8:

            truth_labels = to_onehot(test_labels)
            Y_hat = predicted_labels
            predicted_labels = to_onehot(predicted_labels)
            untrain_class = []
            for i in range(len(mods)):
                if i not in train_class:
                    untrain_class.append(i)
            mods_sort = [mods[i] for i in train_class] + [mods[i] for i in untrain_class]
            if max(Y_hat) >= len(mods_sort):
                new_labels = [mods_sort[i] if i < len(mods_sort) else f'new unknown{i - len(mods_sort) + 1}' for i in
                                range(max(Y_hat) + 1)]
            else:
                new_labels = mods_sort

            print(predicted_semantics.shape, test_labels.shape, new_labels)
            known_semantics = predicted_semantics[test_labels < num_class]
            unknown_semantics = predicted_semantics[test_labels >= num_class]
            mltools.tsne(predicted_semantics, test_labels, mods_sort,
                            f'./results/{version}/tsne/tsne_coef{round(coef, 2)}.jpg', num_known_class=num_class)
            mltools.tsne(known_semantics, test_labels[test_labels < num_class], mods_sort,
                            f'./results/{version}/tsne/tsne_known_coef{round(coef, 2)}.jpg')
            mltools.tsne(unknown_semantics, test_labels[test_labels >= num_class], mods_sort,
                            f'./results/{version}/tsne/tsne_unknown_coef{round(coef, 2)}.jpg')
            confnorm = mltools.calculate_confusion_matrix(truth_labels, predicted_labels, train_class, mods, new_labels)
            mltools.plot_confusion_matrix(confnorm,
                                            xlabels=new_labels,
                                            ylabels=mods_sort,
                                            train_class=train_class,
                                            untrain_class=untrain_class,
                                            save_filename=f'results/{version}/confusion_matrix/ConfusionMatrix_coef{round(coef, 2)}.png')

            y, y_pred = np.array(test_labels), np.array(Y_hat)
            y_pred[y_pred == -1] = np.max(y_pred)
            for u in range(len(mods) - len(train_class)):

                max_uc_pred = 0
                y_pred_uc = list(y_pred[y == (len(train_class) + u)])
                for pred in np.unique(y_pred_uc):
                    if y_pred_uc.count(pred) > max_uc_pred:
                        max_uc_pred = pred
                max_uc_pred_num = y_pred_uc.count(max_uc_pred)
                uc_pred_num = y_pred_uc.count((len(train_class) + u))

                change_num = 0
                if (max_uc_pred_num - uc_pred_num) > 0:
                    for i, _ in enumerate(Y_hat):
                        if y[i] == (len(train_class) + u) and y_pred[i] == max_uc_pred:
                            y_pred[i] = (len(train_class) + u)
                            change_num += 1
                        if change_num == (max_uc_pred_num - uc_pred_num):
                            break

            y[y < num_class] = 1
            y[y >= num_class] = -1
            y_pred_scores = 1 / (1 + np.exp(-np.array(dists_svm)))
            # y_pred[y_pred < num_class] = 1
            # y_pred[y_pred >= num_class] = -1
            fpr, tpr = mltools.plot_roc(
                y,
                y_pred_scores,
                name='FOSSP',
                color='red',
                save_filename=f'results/{version}/roc/ROC_coef{round(coef, 2)}.png'
            )
            if coef == 0.33:
                np.save(f'results/{version}/fpr.npy', fpr)
                np.save(f'results/{version}/tpr.npy', tpr)

            for i, uc in enumerate(untrain_class):
                RESULT_LOGGER(resultlines, "Accuracy(unknown class:{}):{}".format(uc, confnorm[
                    len(train_class) + i, len(train_class) + i]))

            oa = np.trace(confnorm) / confnorm.sum()
            oa_list.append(oa)
            precision_cls = np.diag(confnorm[:len(mods_sort), :len(mods_sort)]) / confnorm[:len(mods_sort),
                                                                                    :len(mods_sort)].sum(axis=1)
            recall_cls = np.diag(confnorm[:len(mods_sort), :len(mods_sort)]) / confnorm[:len(mods_sort),
                                                                                :len(mods_sort)].sum(axis=0)
            f1_cls = (2 * precision_cls * recall_cls) / (precision_cls + recall_cls)
            f1 = np.nanmean(f1_cls)
            # f1 = f1_score(test_labels, Y_hat, average='weighted')
            f1_list.append(f1)

            RESULT_LOGGER(resultlines, "OA: {}".format(oa))
            RESULT_LOGGER(resultlines, f'mean F1-score: {f1}')
            RESULT_LOGGER(resultlines, '')
            f.writelines(resultlines)

            if coef < 1.0:
                coef += 0.02

        plt.figure(1)

        ax = plt.axes()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.xlabel('coefficient')
        plt.plot(coef_list, tkr_list, color='blue', linewidth=1, linestyle="solid", label="true known rate")
        plt.plot(coef_list, tur_list, color='green', linewidth=1, linestyle="solid", label="true unknown rate")
        plt.legend(loc=0)
        plt.grid()
        plt.xlim(0, 1)
        plt.ylim(0, 100)
        plt.title('Effect of Coefficient')
        plt.savefig(f'./results/{version}/coefficient.jpg')
        # plt.show()
        plt.close()
        np.save(f'./results/{version}/coef.npy', coef_list)
        np.save(f'./results/{version}/tkr.npy', tkr_list)
        np.save(f'./results/{version}/tur.npy', tur_list)

        plt.figure(2)

        ax = plt.axes()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.xlabel('coefficient')
        plt.plot(coef_list, [i * 100 for i in oa_list], color='blue', linewidth=1, linestyle="solid",
                 label="overall accuracy")
        plt.plot(coef_list, [i * 100 for i in f1_list], color='green', linewidth=1, linestyle="solid", label="F1-score")
        plt.legend(loc=0)
        plt.grid()
        plt.xlim(0, 1)
        plt.ylim(0, 100)
        plt.title('Effect of Coefficient')
        plt.savefig(f'./results/{version}/oa-f1.jpg')
        # plt.show()
        plt.close()
        np.save(f'./results/{version}/oa.npy', oa_list)
        np.save(f'./results/{version}/f1.npy', f1_list)

    print('{} cluster finished'.format('Evalution zero-shot'))


if __name__ == '__main__':
    dataset.split_unknown()
    train_map, test_map, unknown_test_map = dataset.get_train_test_maps()

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

    if os.path.exists('results/' + version):
        shutil.rmtree('results/' + version)
        os.mkdir('results/' + version)
    else:
        os.mkdir('results/' + version)

    if not os.path.isdir('results/' + version + '/confusion_matrix/'):
        os.mkdir('results/' + version + '/confusion_matrix/')

    if not os.path.isdir('results/' + version + '/tsne/'):
        os.mkdir('results/' + version + '/tsne/')

    if not os.path.isdir('results/' + version + '/roc/'):
        os.mkdir('results/' + version + '/roc/')

    model.to(device)

    for model_path in model_paths:
        print('Loading model from {}'.format(model_path))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        zero_shot_path = 'results/' + version + '/ZSL_' + os.path.split(model_path)[-1][:-4] + '.txt'
        print('With {} epochs training...'.format(checkpoint['epoch'] - 1))

        with torch.no_grad():
            model.eval()

            print('ZSL evaluation')
            cal_acc_evol(train_map, test_map, unknown_test_map, version)

    print('end')
