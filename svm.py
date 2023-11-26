from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from tqdm import tqdm  # 导入tqdm库

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
    y_pred = []
    for i in tqdm(range(v_test.shape[0]), desc='Predicting progress'):  # 使用tqdm显示进度条
        y_pred.append(model.predict(v_test[i]))
    print('Penalty Constant: ', C)
    print('Kernel Function: ', kernel)
    print('Macro-F1: ', f1_score(y_true, y_pred, average='macro') * 100, '%')

    return