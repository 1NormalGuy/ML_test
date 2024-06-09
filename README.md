# 基于感知机、KNN、朴素贝叶斯、SVM的文本分类
**2024年武汉大学国家网络安全学院机器学习实验课程大作业**

本次实验主要研究了**KNN、感知机、朴素贝叶斯和SVM**四种文本分类算法。  

实验数据集为20Newsgroups，其中包含20个类别，共18846个文档。

实验主要从以下几个方面进行：

1. 数据的预处理，包括分词、停用词处理、词频统计等。
2. **KNN**算法的实现，研究不同K值和距离度量的影响。
3. **感知机**算法的实现，包括基础感知机和多层感知机（MLP）。
4. **朴素贝叶斯**算法的实现，研究先验概率和条件概率的计算。
5. **SVM**算法的实现，研究不同参数的影响，包括惩罚参数C和核函数类型。
6. 比较四种算法的分类性能。

## 数据预处理

数据预处理是文本分类的重要步骤，包括以下几个步骤：
- **分词**：将文本划分为单个单词或词组。
- **停用词处理**：移除对分类无用的高频词，如"the", "is", "in"等。
- **词频统计**：统计每个单词在文档中的出现频率。
- **TF-IDF加权**：使用TF-IDF（词频-逆文档频率）对词向量进行加权，提高重要特征的区分度。

## KNN算法

KNN算法的关键参数是K值和距离度量。实验分别测试了K=1, 3, 5, 7, 10, 15的不同取值，发现当K在3-7时，分类效果最好。距离度量使用了L1（曼哈顿距离）和L2（欧氏距离）。

### 代码示例
```python
def knn_test(dataChoice, p, k):
    C = get_class_id(dataChoice + '/v_train')
    trainData, labels = get_data_and_labels(dataChoice + '/v_train')

    t = {}.fromkeys(range(len(C)), 0)
    f = {}.fromkeys(range(len(C)), 0)
    pre = {}.fromkeys(range(len(C)), 0)
    printed=set()
    for i in C:
        if C[i] not in printed:
            print('Test', C[i], '...')
            printed.add(C[i])
        curPath = dataChoice + '/v_test/' + C[i]
        for file in os.listdir(curPath):
            pc = knn_predict(curPath + '/' + file, trainData, labels, C, p, k)
            if pc == i:
                t[i] += 1
            else:
                f[i] += 1
                pre[pc] += 1
    precision = {}
    recall = {}
    f1 = {}
    for i in range(len(C)):
        precision[i] = float(t[i]) / (t[i] + pre[i])
        recall[i] = float(t[i]) / (t[i] + f[i])
        f1[i] = float(2 * precision[i] * recall[i]) / (precision[i] + recall[i])
    return
```

## 感知机算法

感知机算法实现了在线学习，通过迭代更新权重向量和偏置项。除了基础感知机外，本实验还实现了多层感知机（MLP），进一步提升了分类能力。

### 代码示例
```python
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
```

## 朴素贝叶斯算法

朴素贝叶斯算法计算每个类别的先验概率和条件概率，然后通过贝叶斯规则进行分类。实验中，朴素贝叶斯算法在多分类问题中的表现尤为突出。

### 代码示例
```python
for c in Dic:
    Nc = nm.get_sum_words(pathLoadChoice + '/' + Dic[c] + '.txt', 0)
    prior[c] = Nc / N

for c in Dic.keys():
    score.append(math.log(prior[str(c)]))
    for t in V:
        if t in vd.keys():
            score[c] += math.log(condprob[int(t)][c])
```

## SVM算法

实验使用了sklearn中的SVM实现，主要测试了不同类别、不同惩罚参数C、不同核函数的影响。实验结果表明SVM在分类性能上优于其他算法。

### 代码示例
```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import SVC

def svm_test(category, C, kernel):
    train = fetch_20newsgroups(subset='train', categories=category)
    test = fetch_20newsgroups(subset='test', categories=category)

    vectorizer = TfidfVectorizer()
    v_train = vectorizer.fit_transform(train.data)
    v_test = vectorizer.transform(test.data)

    model = SVC(C=C, kernel=kernel)
    model.fit(v_train, train.target)

    y_true = test.target
    y_pred = model.predict(v_test)

    return f1_score(y_true, y_pred, average='macro')
```

## 结论

- KNN算法效果与K值和距离度量有关。当K值在3-7之间时，分类效果最好。L2距离度量通常优于L1距离度量。
- 感知机和SVM算法更适用于二分类问题，其分类准确率可以达到95\%以上。SVM通过最大化类别间隔，进一步提高了分类的鲁棒性和准确性。
- 朴素贝叶斯算法更适用于多分类问题，其分类准确率可以达到80\%以上。通过计算先验概率和条件概率，朴素贝叶斯能够有效处理高维数据。
- SVM分类效果优于其他算法，在不同参数设置下表现出色，特别是使用线性核函数时效果最佳。
- 经过TF-IDF加权的词向量分类效果优于简单词频向量，能够提高模型的区分能力和准确性。

