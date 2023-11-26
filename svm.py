from sklearn.datasets import fetch_20newsgroups  # 导入sklearn库中的新闻组数据集
from sklearn.feature_extraction.text import TfidfVectorizer  # 导入TF-IDF向量化工具
from sklearn.metrics import f1_score  # 导入F1分数计算工具
from sklearn.svm import SVC  # 导入支持向量机分类器

# 定义一个函数，用于测试SVM模型
def svm_test(category, C, kernel):
    print('\r' + '==================== SVM ====================')
    train = fetch_20newsgroups(subset='train', categories=category)  
    test = fetch_20newsgroups(subset='test', categories=category) 

    vectorizer = TfidfVectorizer()  
    v_train = vectorizer.fit_transform(train.data)  
    v_test = vectorizer.transform(test.data)  

    model = SVC(C=C, kernel=kernel)  
    model.fit(v_train, train.target)  

    y_true = test.target  
    y_pred = model.predict(v_test) 
    print('Penalty Constant: ', C) 
    print('Kernel Function: ', kernel)  
    print('Macro-F1: ', f1_score(y_true, y_pred, average='macro') * 100, '%') 

    return