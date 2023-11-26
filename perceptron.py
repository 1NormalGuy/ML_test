import numpy as np
from pre_processing import normalization as nm
import os
from prettytable import PrettyTable
import math
from tqdm import tqdm

# 获取向量列表
def get_list_of_vectors():
    path = 'dataof2/v_train' 
    tempList = [] 
    for dir in os.listdir(path):
        curPath = os.path.join(path, dir) 
        for file in os.listdir(curPath):
            vec = nm.txt_to_dic(os.path.join(curPath, file))  
            tempList.append(vec)  
    return tempList  



# 获取逆文档频率（IDF）
def get_idf():
    tempList = get_list_of_vectors()  
    vAll = nm.txt_to_dic('data_2/allDic.txt')  
    vIdf = {}  
    len_D = len(tempList) 
    for v in vAll:
        df = 0 
        for d in tempList:
            if v in d.keys():
                df += 1
        vIdf[v] = math.log(float(len_D)/(df + 1), 10)
    return vIdf 

# 打印混淆矩阵
def print_confusion_matrix(categories, tp, tn, fp, fn):
    table = PrettyTable(['Actual \\ Predict', categories[0], categories[1]])  
    table.add_row([categories[0], tn, fp])  
    table.add_row([categories[1], fn, tp])  
    return table  

# 获取文档向量
def get_d_array(path, vAll):
    dic = nm.txt_to_dic(path)  
    l = np.zeros(len(vAll))  
    for v in vAll:
        if v in dic.keys():
            l[int(v)] = int(dic[v])
    return l  

# 定义一个函数，用于获取数据和标签
def get_data_and_labels(dataChoice, vAll, categories):
    path = dataChoice + '/v_test'  
    data = []  
    labels = []  
    classDic = {1: categories[0], -1: categories[1]}  
    for dir in os.listdir(path):
        curPath = os.path.join(path, dir)  
        if dir == classDic[1]:
            for file in os.listdir(curPath):
                data.append(get_d_array(os.path.join(curPath, file), vAll))
                labels.append(1)
        elif dir == classDic[-1]:
            for file in os.listdir(curPath):
                data.append(get_d_array(os.path.join(curPath, file), vAll))
                labels.append(-1)
    return data, labels  


# 定义一个函数，用于训练感知机模型
def perceptron_train(dataChoice, categories, max_iter, a):
    vAll = nm.txt_to_dic(dataChoice + '/allDic.txt')  
    data, labels = get_data_and_labels(dataChoice, vAll, categories)  
    w = np.zeros(len(vAll))  
    b = 0  
    for iter in range(max_iter):
        for i in range(len(data)):
            x = data[i] 
            y = labels[i] 
            if y * (np.dot(x, w) + b) <= 0:
                delta = np.multiply(a*y, x)  
                w = np.add(w, delta) 
                b += a*y 
    return w, b, vAll  

# 预测文档的类别
def perceptron_predict(w, b, d):
    return np.sign(np.dot(d, w) + b)  

# 测试感知机模型
def perceptron_test(dataChoice, categories, max_iter, a):
    print('\r'+'==================== Perceptron ====================')
    w, b, vAll = perceptron_train(dataChoice, categories, max_iter, a)  
    tp = tn = fp = fn = 0  
    classDic = {1: categories[0], -1: categories[1]}  
    for iter in tqdm(range(max_iter), desc='Training progress'):
        for cname in os.listdir(dataChoice + '/v_test'):
            if cname == classDic[1]:
                dir_path = os.path.join(dataChoice + '/v_test', cname)
                for file in os.listdir(dir_path):
                    d = os.path.join(dir_path, file)  
                    result = perceptron_predict(w, b, get_d_array(d, vAll))  
                    if result == 1:
                        tp += 1
                    else:
                        fn += 1
            else:
                dir_path = os.path.join(dataChoice + '/v_test', cname)  
                for file in os.listdir(dir_path):
                    d = os.path.join(dir_path, file) 
                    result = perceptron_predict(w, b, get_d_array(d, vAll))  
                    if result == -1:
                        tn += 1
                    else:
                        fp += 1
    print('\r')
    print('Iteration Times: ', max_iter)  
    print('Learning Rate: ', a)  
    print("Confusion matrix: ")  
    print(print_confusion_matrix([classDic[-1], classDic[1]], tp, tn, fp, fn))
    print("Precision: ", float(tp) / (tp + fp) * 100, "%")  
    print("Recall: ", float(tp) / (tp + fn) * 100, "%")  
    print('F1-score: ', float(2*tp) / (2*tp + fp + fn) * 100, "%")  