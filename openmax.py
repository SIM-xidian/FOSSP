import libmr
import torch
import numpy as np
import scipy.spatial.distance as spd
from torch.utils.data import TensorDataset, DataLoader
from config_resnet import *
import mltools
from sklearn.metrics import f1_score


def RESULT_LOGGER(result_list, message):
    result_list.append('{}\n'.format(message))
    print(message)


def compute_channel_distances(mavs, features, eu_weight=0.5):
    """
    Input:
        mavs (channel, C)
        features: (N, channel, C)
    Output:
        channel_distances: dict of distance distribution from MAV for each channel.
    """
    # eu_dists: euclidean distance; cos_dists: cosine distance; eucos_dists: euclidean distance and cosine distance
    eucos_dists, eu_dists, cos_dists = [], [], []
    for channel, mcv in enumerate(mavs):  # Compute channel specific distances
        if len(mcv) == 0:
            eu_dists.append([])
            eucos_dists.append([])
            cos_dists.append([])
            continue
        eu_dists.append([spd.euclidean(mcv, feat[channel]) for feat in features])
        cos_dists.append([spd.cosine(mcv, feat[channel]) for feat in features])
        eucos_dists.append([spd.euclidean(mcv, feat[channel]) * eu_weight +
                            spd.cosine(mcv, feat[channel]) for feat in features])

    return {'eucos': np.array(eucos_dists), 'cosine': np.array(cos_dists), 'euclidean': np.array(eu_dists)}


def compute_mavs_and_dists(num_classes, train_dataloader, device, model):
    try:
        dummy_scores = [[] for _ in range(num_classes)]
        with torch.no_grad():
            for batch_idx, (batch_inputs, batch_labels) in enumerate(train_dataloader):
                batch_inputs, batch_labels = batch_inputs.float().to(device), batch_labels.long().to(device)

                logits = model(batch_inputs)
                # print(logits.shape)
                for j in range(logits.shape[0]):
                    if np.argmax(logits[j].cpu().detach().numpy(), axis=0) == batch_labels[j]:
                        dummy_scores[batch_labels[j]].append(logits[j].unsqueeze(dim=0).unsqueeze(dim=0))

        scores = []
        for score_list in dummy_scores:
            if score_list:
                scores.append(torch.cat(score_list).cpu().numpy())
            else:
                scores.append([])

        mavs = []
        for score in scores:
            if len(score) > 0:
                mavs.append(np.mean(score, axis=0))
            else:
                mavs.append([])

        dists = [compute_channel_distances(mcv, score) for mcv, score in zip(mavs, scores)]

        return mavs, dists
    except Exception as e:
        print(e)
        print(e.__traceback__.tb_lineno)


def weibull(means, dists, categories, tailsize=20, distance_type='eucos'):
    """
    Input:
        means (C, channel, C)
        dists (N_c, channel, C) * C
    Output:
        weibull_model : Perform EVT based analysis using tails of distances and save
                        weibull model parameters for re-adjusting softmax scores
    """
    weibull_model = {}
    for mean, dist, category_name in zip(means, dists, categories):

        weibull_model[category_name] = {}
        weibull_model[category_name]['distances_{}'.format(distance_type)] = dist[distance_type]
        weibull_model[category_name]['mean_vec'] = mean
        weibull_model[category_name]['weibull_model'] = []
        if len(mean) == 0 or len(dist) == 0:
            continue
        for channel in range(mean.shape[0]):
            mr = libmr.MR()
            tailtofit = np.sort(dist[distance_type][channel, :])[-tailsize:]
            mr.fit_high(tailtofit, len(tailtofit))
            weibull_model[category_name]['weibull_model'].append(mr)

    return weibull_model


def query_weibull(category_name, weibull_model, distance_type='eucos'):
    return [weibull_model[category_name]['mean_vec'],
            weibull_model[category_name]['distances_{}'.format(distance_type)],
            weibull_model[category_name]['weibull_model']]


def calc_distance(query_score, mcv, eu_weight, distance_type='eucos'):
    global query_distance
    if distance_type == 'eucos':
        query_distance = spd.euclidean(mcv, query_score) * eu_weight + \
                         spd.cosine(mcv, query_score)
    elif distance_type == 'euclidean':
        query_distance = spd.euclidean(mcv, query_score)
    elif distance_type == 'cosine':
        query_distance = spd.cosine(mcv, query_score)
    else:
        print("distance type not known: enter either of eucos, euclidean or cosine")
    return query_distance


def compute_openmax_prob(scores, scores_u):
    prob_scores, prob_unknowns = [], []
    for s, su in zip(scores, scores_u):
        channel_scores = np.exp(s)
        channel_unknown = np.exp(np.sum(su))

        total_denom = np.sum(channel_scores) + channel_unknown
        prob_scores.append(channel_scores / total_denom)
        prob_unknowns.append(channel_unknown / total_denom)

    # Take channel mean
    scores = np.mean(prob_scores, axis=0)
    unknowns = np.mean(prob_unknowns, axis=0)
    modified_scores = scores.tolist() + [unknowns]
    return modified_scores


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def openmax(weibull_model, categories, input_score, eu_weight, alpha=10, distance_type='eucos'):
    """Re-calibrate scores via OpenMax layer
    Output:
        openmax probability and softmax probability
    """
    num_classes = len(categories)
    # select the right label's every channel score which are max score of scores.
    ranked_list = input_score.argsort().ravel()[::-1][:alpha]
    # the labels are argsorted highly, they will obtain huge weights.
    # alpha_weight: [3/3, 2/3, 1/3]
    alpha_weights = [((alpha + 1) - i) / float(alpha) for i in range(1, alpha + 1)]
    omega = np.zeros(num_classes)
    omega[ranked_list] = alpha_weights
    # we will change the scores and store uncertainty scores
    scores, scores_u = [], []
    for channel, input_score_channel in enumerate(input_score):
        score_channel, score_channel_u = [], []
        for c, category_name in enumerate(categories):
            mav, dist, model = query_weibull(category_name, weibull_model, distance_type)
            channel_dist = calc_distance(input_score_channel, mav[channel], eu_weight, distance_type)
            wscore = model[channel].w_score(channel_dist)
            modified_score = input_score_channel[c] * (1 - wscore * omega[c])
            score_channel.append(modified_score)
            score_channel_u.append(input_score_channel[c] - modified_score)

        scores.append(score_channel)
        scores_u.append(score_channel_u)

    scores = np.asarray(scores)
    scores_u = np.asarray(scores_u)

    openmax_prob = np.array(compute_openmax_prob(scores, scores_u))
    softmax_prob = softmax(np.array(input_score.ravel()))
    return openmax_prob, softmax_prob


if __name__ == '__main__':

    if not os.path.isdir('results/' + version + '_openmax'):
        os.mkdir('results/' + version + '_openmax')

    if not os.path.isdir('results/' + version + '_openmax/confusion_matrix/'):
        os.mkdir('results/' + version + '_openmax/confusion_matrix/')

    if not os.path.isdir('results/' + version + '_openmax/roc/'):
        os.mkdir('results/' + version + '_openmax/roc/')

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
    mavs, dists = compute_mavs_and_dists(num_classes=num_class, train_dataloader=train_dataloader, device=device,
                                         model=model)

    val_dataset = TensorDataset(torch.from_numpy(test_samples), torch.from_numpy(test_labels))
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1024, shuffle=False)

    scores, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch_inputs, batch_labels in val_dataloader:
            batch_inputs = batch_inputs.float().cuda()
            batch_labels = batch_labels.long().cuda()
            logits = model(batch_inputs)

            scores.append(logits)
            labels.append(batch_labels)

    scores = torch.cat(scores, dim=0).detach().cpu().numpy()
    labels = torch.cat(labels, dim=0).detach().cpu().numpy()
    scores = np.array(scores)[:, np.newaxis, :]
    labels = np.array(labels)

    categories = list(range(0, num_class))
    weibull_model = weibull(means=mavs, dists=dists, categories=categories, tailsize=3,
                            distance_type='euclidean')

    all_mcv_filled = True
    for mcv in mavs:
        if len(mcv) == 0:
            all_mcv_filled = False
    if all_mcv_filled:
        pred_softmax, pred_softmax_threshold, pred_openmax = [], [], []
        score_softmax, score_openmax = [], []
        for score in scores:
            so, ss = openmax(weibull_model, categories, score, 0.5, 3,
                             "euclidean")  # openmax_prob, softmax_prob
            pred_softmax.append(np.argmax(ss))
            pred_softmax_threshold.append(np.argmax(ss) if np.max(ss) >= 0.8 else num_class)
            pred_openmax.append(np.argmax(so) if np.max(so) >= 0.8 else num_class)
            score_softmax.append(ss)
            score_openmax.append(so)

        untrain_class = []
        for i in range(len(mods)):
            if i not in train_class:
                untrain_class.append(i)
        mods_sort = [mods[i] for i in train_class] + [mods[i] for i in untrain_class]
        new_labels = [mods[i] for i in train_class]
        new_labels.append('unknown')

        pred_openmax = np.array(pred_openmax)
        confnorm = mltools.calculate_confusion_matrix1(labels, pred_openmax, mods)
        mltools.plot_confusion_matrix1(confnorm,
                                       xlabels=new_labels,
                                       ylabels=mods_sort,
                                       train_class=train_class,
                                       untrain_class=untrain_class,
                                       save_filename=f'results/{version}_openmax/confusion_matrix/ConfusionMatrix.png')

        y, y_pred = labels.copy(), pred_openmax.copy()
        y[y < num_class] = 1
        y[y >= num_class] = -1
        # y_pred[y_pred < num_class] = 1
        # y_pred[y_pred >= num_class] = -1
        fpr, tpr = mltools.plot_roc(
            y,
            [max(i) for i in score_openmax],
            name='OpenMax',
            color='yellow',
            save_filename=f'results/{version}_openmax/roc/ROC.png'
        )

        np.save(f'results/{version}_openmax/fpr.npy', fpr)
        np.save(f'results/{version}_openmax/tpr.npy', tpr)

        zero_shot_path = 'results/' + version + '_openmax/' + os.path.split(model_path)[-1][:-4] + '.txt'
        with open(zero_shot_path, 'w') as f:
            resultlines = []
            conf = confnorm[:num_class, :num_class]
            known_oa = np.trace(conf) / conf.sum()
            precision_cls = np.diag(conf) / conf.sum(axis=1)
            recall_cls = np.diag(conf) / conf.sum(axis=0)
            f1_cls = (2 * precision_cls * recall_cls) / (precision_cls + recall_cls)
            known_f1 = np.nanmean(f1_cls)

            for i in range(num_class):
                RESULT_LOGGER(resultlines, "Accuracy(class:{}): {}".format(train_class[i], conf[i, i]))
            RESULT_LOGGER(resultlines, 'Known OA: {}'.format(known_oa))
            RESULT_LOGGER(resultlines, 'Known F1-score: {}'.format(known_f1))
            RESULT_LOGGER(resultlines, '')
            for i in range(len(unknown_class)):
                RESULT_LOGGER(resultlines, 'Accuracy(unknown class:{}): {}'.format(list(unknown_class)[i], confnorm[i + num_class, num_class]))

            o_conf = np.zeros([conf.shape[0] + 1, conf.shape[1] + 1])
            for i in range(pred_openmax.shape[0]):
                m = labels[i]
                if m < num_class:
                    n = pred_openmax[i]
                    o_conf[m, n] += 1
                else:
                    n = pred_openmax[i]
                    o_conf[num_class, n] += 1
            o_conf[:, :] = o_conf[:, :].astype('float') / o_conf[:, :].sum(axis=1)[:, np.newaxis]
            o_conf = np.around(o_conf, decimals=2)

            oa = np.trace(o_conf) / o_conf.sum()
            precision_cls = np.diag(o_conf) / o_conf.sum(axis=1)
            recall_cls = np.diag(o_conf) / o_conf.sum(axis=0)
            f1_cls = (2 * precision_cls * recall_cls) / (precision_cls + recall_cls)
            f1 = np.nanmean(f1_cls)
            RESULT_LOGGER(resultlines, "OA: {}".format(oa))
            RESULT_LOGGER(resultlines, f'mean F1-score: {f1}')
            f.writelines(resultlines)
