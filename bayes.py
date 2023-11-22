# 导入所需的库
from pre_processing import normalization as nm 
import os  
import math  
import numpy as np  
from prettytable import PrettyTable  

# 获取类别名
def get_class_id(path):
    Dic = {}  
    i = 0  
    for cName in os.listdir(path):  
        Dic[i] = cName[:-4]  # 将文件名去掉.txt
        i += 1  
    return Dic  

# 打印精确度、召回率和F1分数矩阵
def print_prf_matrix(Dic, precision, recall, f1):
    table = PrettyTable(['Class', 'Precision', 'Recall', 'F1-score'])  
    for i in range(len(Dic)): 
        table.add_row([Dic[i], precision[i], recall[i], f1[i]])  
    return table 

# 将每个类别目录转储为一个向量
def class_to_txt(dataChoice):
    pathLoadChoice = dataChoice + '/v_train'  
    pathStoreChoice = dataChoice + '/bayes/v_class'  
    if not os.path.exists(pathStoreChoice):  
        os.makedirs(pathStoreChoice)  
    for dir in os.listdir(pathLoadChoice):  
        c_dir = pathLoadChoice + '/' + dir  
        v = nm.merge_all_vec_in_sort(c_dir)  
        nm.dic_to_txt(v, pathStoreChoice + '/' + dir + '.txt') 
    return

# 将先验概率转储为文本
def prior_to_txt(dataChoice, Dic, N):
    pathLoadChoice = dataChoice + '/bayes/v_class'  
    pathStoreChoice = dataChoice + '/bayes/prior.txt'  
    prior = {}  # 创建一个空字典用于存储先验概率
    for c in Dic:  # 遍历所有类别
        Nc = nm.get_sum_words(pathLoadChoice + '/' + Dic[c] + '.txt', 0)  
        prior[c] = Nc/N 
    nm.dic_to_txt(prior, pathStoreChoice)  
    return

# 计算并保存条件概率
def condprob_to_npy(dataChoice, Dic, V, lamb):
    NV = len(V)  
    condprob = [[0 for i in range(len(Dic))] for j in range(NV)]  
    for c in Dic: 
        vc_path = dataChoice + '/bayes/v_class' + '/' + Dic[c] + '.txt'  
        ck = nm.get_sum_words(vc_path, 0)  
        vc = nm.txt_to_dic(vc_path)  
        for t in V:  
            try:
                Tct = vc[t]  
            except:
                Tct = 0  
            condprob[int(t)][c] = (int(Tct) + lamb) / (ck + NV*lamb) 
    np.save(dataChoice + '/bayes/condprob.npy', np.array(condprob))  
    return

# 训练朴素贝叶斯模型
def naive_bayes_train(dataChoice, lamb):
    class_to_txt(dataChoice) 
    print('Finished getting vectors of classes.')  
    Dic = get_class_id(dataChoice + '/bayes/v_class') 
    N = nm.get_sum_words(dataChoice + '/bayes/v_class', 0)  
    V = nm.txt_to_dic(dataChoice + '/allDic.txt') 
    prior_to_txt(dataChoice, Dic, N) 
    print('Finished getting prior probability.')  
    condprob_to_npy(dataChoice, Dic, V, lamb)  
    print('Finished getting conditional probability')  
    return

# 使用朴素贝叶斯模型进行预测
def naive_bayes_predict(d, prior, condprob, Dic, V):
    vd = nm.txt_to_dic(d)  
    score = []  
    for c in Dic.keys(): 
        score.append(math.log(prior[str(c)]))  
        for t in V: 
            if t in vd.keys(): 
                score[c] += math.log(condprob[int(t)][c])  
    max_index = score.index(max(score))  
    return max_index 

# 使用朴素贝叶斯模型进行测试
def naive_bayes_test(dataChoice, lamb):
    print('\r' + '==================== Naive Bayes ====================')  
    naive_bayes_train(dataChoice, lamb)  
    prior = nm.value_to_float(nm.txt_to_dic(dataChoice + '/bayes/prior.txt'))  
    condprob = (np.load(dataChoice + '/bayes/condprob.npy', allow_pickle=True)).tolist()  
    Dic = get_class_id(dataChoice + '/bayes/v_class') 
    V = nm.txt_to_dic(dataChoice + '/allDic.txt')  
    dir = dataChoice + '/v_test'  

    true = {}.fromkeys(range(len(Dic)), 0) 
    false = {}.fromkeys(range(len(Dic)), 0)  
    predict= {}.fromkeys(range(len(Dic)), 0)  
    for i in Dic:  
        print(i+1,':','Test', Dic[i], '...')
        cur_path = dir + '/' + Dic[i]
        for file in os.listdir(cur_path):
            pc = naive_bayes_predict(cur_path + '/' + file, prior, condprob, Dic, V)
            if pc == i:
                true[i] += 1
            else:
               false[i] += 1 # recall(i) = true[i]/(true[i]+f[i])
               predict[pc] += 1 # precision(i) = true[i]/(true[i] + p[i])
    precision = {}
    recall = {}
    f1 = {}
    for i in range(len(Dic)):
        precision[i] = float(true[i]) / (true[i] + predict[i])
        recall[i] = float(true[i]) / (true[i]+false[i])
        f1[i] = float(2*precision[i]*recall[i]) / (precision[i] + recall[i])
    print('Lambda: ', lamb)
    print(print_prf_matrix(Dic, precision, recall, f1))
    print('Macro-F1: ', float(sum(f1.values())) / len(Dic) * 100, '%')

    return
