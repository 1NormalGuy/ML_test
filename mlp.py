from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm  # 导入tqdm库

def mlp_test(category, hidden_layer_sizes, activation, solver, max_iter):
    print('\r' + '==================== MLP ====================')
    train = fetch_20newsgroups(subset='train', categories=category)
    test = fetch_20newsgroups(subset='test', categories=category)

    vectorizer = TfidfVectorizer()
    v_train = vectorizer.fit_transform(train.data)
    v_test = vectorizer.transform(test.data)

    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, max_iter=max_iter)
    model.fit(v_train, train.target)

    y_true = test.target
    y_pred = []
    for i in tqdm(range(v_test.shape[0]), desc='Predicting progress'):  # 使用tqdm显示进度条
        y_pred.append(model.predict(v_test[i]))
    print('Hidden Layer Sizes: ', hidden_layer_sizes)
    print('Activation Function: ', activation)
    print('Solver: ', solver)
    print('Max Iterations: ', max_iter)
    print('Macro-F1: ', f1_score(y_true, y_pred, average='macro') * 100, '%')

    return

# 示例调用
categories = ['alt.atheism', 'soc.religion.christian']
mlp_test(categories, hidden_layer_sizes=(500,), activation='relu', solver='adam', max_iter=200)
