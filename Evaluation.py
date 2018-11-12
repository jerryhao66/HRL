import numpy as np
import heapq
import logging
from time import time
import math

def predict_with_atteionts(model, test_user_input, test_num_idx, test_item_input, test_labels, test_batch_num):
    hits5, ndcgs5, hits10, ndcgs10, maps, mrrs, attentions, losses = [], [], [], [], [], [], [], []
    for batch in range(test_batch_num):
        t_u = test_user_input[batch]
        t_num = test_num_idx[batch]
        t_i = test_item_input[batch]
        t_label = test_labels[batch]
        predictions, attention, loss = model.predict_with_atteionts(t_u, np.reshape(t_num,(-1,1)), np.reshape(t_i,(-1,1)), np.reshape(t_label,(-1,1)))
        map_item_score = {t_i[i]: predictions[i] for i in range(len(t_i))}
        gtItem = t_i[-1]
        ranklist5 = heapq.nlargest(5, map_item_score, key=map_item_score.get)
        ranklist10 = heapq.nlargest(10, map_item_score, key=map_item_score.get)
        ranklist100 = heapq.nlargest(100, map_item_score, key=map_item_score.get)
        hr5 = getHitRatio(ranklist5, gtItem)
        ndcg5 = getNDCG(ranklist5, gtItem)
        hr10 = getHitRatio(ranklist10, gtItem)
        ndcg10 = getNDCG(ranklist10, gtItem)
        ap = getAP(ranklist100, gtItem)
        mrr = getMRR(ranklist100, gtItem)
        hits5.append(hr5)
        ndcgs5.append(ndcg5)
        hits10.append(hr10)
        ndcgs10.append(ndcg10)
        maps.append(ap)
        mrrs.append(mrr)
        losses.append(loss)
        attentions.append(attention)

    final_hr5, final_ndcg5, final_hr10, final_ndcg10, final_map, final_mrr, final_test_loss = np.array(hits5).mean(), np.array(ndcgs5).mean(), np.array(hits10).mean(),np.array(ndcgs10).mean(), np.array(maps).mean(), np.array(mrrs).mean(), np.array(losses).mean()
    return (final_hr5, final_ndcg5, final_hr10, final_ndcg10, final_map, final_mrr, final_test_loss, attentions)

def eval_rating(model, test_user_input, test_num_idx, test_item_input, test_labels, test_batch_num):
    hits5, ndcgs5, hits10, ndcgs10, maps, mrrs, losses = [], [], [], [], [], [], []
    for batch in range(test_batch_num):
        t_u = test_user_input[batch]
        t_num = test_num_idx[batch]
        t_i = test_item_input[batch]
        t_label = test_labels[batch]
        predictions, loss = model.predict(t_u, np.reshape(t_num,(-1,1)), np.reshape(t_i,(-1,1)), np.reshape(t_label,(-1,1)))
        map_item_score = {t_i[i]: predictions[i] for i in range(len(t_i))}            
        gtItem = t_i[-1]
        ranklist5 = heapq.nlargest(5, map_item_score, key=map_item_score.get)
        ranklist10 = heapq.nlargest(10, map_item_score, key=map_item_score.get)
        ranklist100 = heapq.nlargest(100, map_item_score, key=map_item_score.get)
        hr5 = getHitRatio(ranklist5, gtItem)
        ndcg5 = getNDCG(ranklist5, gtItem)
        hr10 = getHitRatio(ranklist10, gtItem)
        ndcg10 = getNDCG(ranklist10, gtItem)
        ap = getAP(ranklist100, gtItem)
        mrr = getMRR(ranklist100, gtItem)
        hits5.append(hr5)
        ndcgs5.append(ndcg5)
        hits10.append(hr10)
        ndcgs10.append(ndcg10)
        maps.append(ap)
        mrrs.append(mrr)
        losses.append(loss)

    final_hr5, final_ndcg5, final_hr10, final_ndcg10, final_map, final_mrr, final_test_loss = np.array(hits5).mean(), np.array(ndcgs5).mean(), np.array(hits10).mean(),np.array(ndcgs10).mean(), np.array(maps).mean(), np.array(mrrs).mean(), np.array(losses).mean()
    return (final_hr5, final_ndcg5, final_hr10, final_ndcg10, final_map, final_mrr, final_test_loss)

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0


def getAP(ranklist, gtItem):
    hits = 0
    sum_precs = 0
    for n in range(len(ranklist)):
        if ranklist[n] == gtItem:
            hits += 1
            sum_precs += hits / (n + 1.0)
    if hits > 0:
        return sum_precs / 1
    else:
        return 0


def getMRR(ranklist, gtItem):
    for index, item in enumerate(ranklist):
        if item == gtItem:
            return 1.0 / (index + 1.0)
    return 0