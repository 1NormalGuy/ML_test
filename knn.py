import numpy as np
from collections import Counter
from pre_processing import normalization as nm
import os
from prettytable import PrettyTable
from tqdm import tqdm


def get_class_id(path):
    C = {}
    i = 0
    for cName in os.listdir(path):
        C[i] = cName
        i += 1
    return C


def print_prf_matrix(C, precision, recall, f1):
    table = PrettyTable(['\\', 'Precision', 'Recall', 'F1-score'])
    for i in range(len(C)):
        table.add_row([C[i], precision[i], recall[i], f1[i]])
    return table


def get_vd(path):
    d = nm.txt_to_dic(path)
    l = []
    for key, value in d.items():
        l.append([int(key), int(value)])
    return l


def get_data_and_labels(path):
    data = []
    labels = []
    C = get_class_id(path)
    for dir in os.listdir(path):
        curPath = os.path.join(path, dir)
        for i in range(len(C)):
            if dir == C[i]:
                for file in os.listdir(curPath):
                    data.append(get_vd(os.path.join(curPath, file)))
                    labels.append(i)
    return data, labels


def get_dis(d, t, p):
    dis = 0
    p1 = p2 = 0
    len_d = len(d)
    len_t = len(t)
    while p1 < len_d and p2 < len_t:
        if d[p1][0] == t[p2][0]:
            dis += abs(d[p1][1]-t[p2][1]) ** p
            p1 += 1
            p2 += 1
        elif d[p1][0] < t[p2][0]:
            dis += d[p1][1] ** p
            p1 += 1
        else:
            dis += t[p2][1] ** p
            p2 += 1
    while p1 < len_d:
        dis += d[p1][1] ** p
        p1 += 1
    while p2 < len_t:
        dis += t[p2][1] ** p
        p2 += 1
    dis = dis ** 1/p
    return dis


# 构建KNN分类器
def knn_predict(d_path, trainData, labels, C, p, k):
    d = get_vd(d_path)
    dis = []
    for i in range(len(trainData)):
        dis.append(get_dis(d, trainData[i], p))

    dis = np.array(dis)

    sortedIndex = dis.argsort()  
    sortedLabels = []
    for i in range(k):
        c = labels[sortedIndex[i]]  
        sortedLabels.append(c)
    counter = Counter(sortedLabels)
    maxC = counter.most_common(1)[0][0]
    return maxC


def knn_test(dataChoice, p, k):
    print('\r' + '==================== KNN ====================')
    C = get_class_id(dataChoice + '/v_train')
    trainData, labels = get_data_and_labels(dataChoice + '/v_train')

    t = {}.fromkeys(range(len(C)), 0)
    f = {}.fromkeys(range(len(C)), 0)
    pre = {}.fromkeys(range(len(C)), 0)
    printed=set()
    for i in tqdm(C, desc='Test progress'):  
        if C[i] not in printed:
            print('Test', C[i], '...')
            printed.add(C[i])
        curPath = dataChoice + '/v_test/' + C[i]
        for file in os.listdir(curPath):
            pc = knn_predict(curPath + '/' + file, trainData, labels, C, p, k)
            if pc == i:
                t[i] += 1
            else:
                f[i] += 1  # recall(i) = t[i]/(t[i]+f[i])
                pre[pc] += 1  # precision(i) = t[i]/(t[i] + pre[i])
    precision = {}
    recall = {}
    f1 = {}
    for i in range(len(C)):
        precision[i] = float(t[i]) / (t[i] + pre[i])
        recall[i] = float(t[i]) / (t[i] + f[i])
        f1[i] = float(2 * precision[i] * recall[i]) / (precision[i] + recall[i])
    print('P of Minkowski Distance: ', p)
    print('K of Knn: ', k)
    print(print_prf_matrix(C, precision, recall, f1))
    print('Macro-F1: ', float(sum(f1.values())) / len(C) * 100, '%')
    return
   