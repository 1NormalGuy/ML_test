from nltk.tokenize import word_tokenize
import re
import os
from collections import Counter

with open("pre_processing/stop_words.utf8", encoding='utf-8') as f:
    stopwordList = f.read().splitlines()


def get_list_of_text(fileName):
    with open(fileName, encoding='gb18030', errors='ignore') as f:
        text = f.read().splitlines()
    return text


def tokenize_text(text):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if bool(re.search('[a-z]', token))]
    return tokens


def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopwordList]
    return filtered_tokens


def tokenize_document(document):
    tokenizedDocument = []
    for line in document:
        line = remove_stopwords(line)
        tokenizedDocument += line
    return tokenizedDocument


def list_to_dic(list):
    dic = {}
    for i in range(len(list)):
        dic[i] = list[i]
    return dic


def get_all_words(dir, wordsList):
    for fileName in os.listdir(dir):
        curPath = dir + '/' + fileName
        if os.path.isdir(curPath):
            print('loading ' + curPath + '...')
            tmp = get_all_words(curPath, wordsList)
            wordsList += tmp
        else:
            doc = tokenize_document(get_list_of_text(curPath))
            wordsList += doc
    return wordsList


def get_dictionary(dir):
    wordsList = []
    for dirName in os.listdir(dir):
        curPath = dir + '/' + dirName
        print('loading ' + curPath + '...')
        for fileName in os.listdir(curPath):
            doc = tokenize_document(get_list_of_text(curPath + '/' + fileName))
            wordsList += doc
    dic_list = list(set(wordsList))
    dic = list_to_dic(dic_list)
    print('Finished getting dictionary.')
    return dic


def get_id(targetWord, dic):
    for id, word in dic.items():
        if word == targetWord:
            wordId = id
            break
    return wordId


def sorted_vector(vec):
    intKeys = []
    for i in range(len(list(vec.keys()))):
        intKeys.append(int(list(vec.keys())[i]))
    sortedKeys = sorted(intKeys)
    sortedVec = {}
    for i in sortedKeys:
        sortedVec[i] = vec[str(i)]
    return sortedVec


def get_sorted_vector(doc, dic_all):
    doc = tokenize_document(get_list_of_text(doc))
    dicDoc = {}
    for word in set(doc):
        if word in dic_all.values():
            dicDoc[get_id(word, dic_all)] = doc.count(word)
    dicDoc = sorted_vector(dicDoc)
    return dicDoc


def dic_to_txt(dic, path):
    with open(path, 'w+', encoding='utf-8') as f:
        for key, value in dic.items():
            # if isinstance(key, int):
            key = str(key)
            # if isinstance(value, int):
            value = str(value)
            f.write('<' + key + ',' + value + '>')
            f.write('\r')
    return f


def txt_to_dic(txt):
    dic = {}
    with open(txt, 'r', encoding='utf-8') as f:
        tuples = f.read().splitlines()
    for tuple in tuples:
        key = (re.search('<(.*?),', tuple)).group(1)
        val = (re.search(',(.*?)>', tuple)).group(1)
        dic[key] = val
    return dic


def value_to_int(dic):
    for key, value in dic.items():
        dic[key] = int(dic[key])
    return dic


def value_to_float(dic):
    for key, value in dic.items():
        dic[key] = float(dic[key])
    return dic

def merge_all_vec_in_sort(path):
    dic = Counter({})
    for vec in os.listdir(path):
        vec = os.path.join(path, vec)
        dic += Counter(value_to_int(txt_to_dic(vec)))
    dic = sorted_vector(dic)
    return dic


def get_sum_words(path, n):
    if os.path.isfile(path):
        vec = txt_to_dic(path)
        for freq in vec.values():
            n += int(freq)
        return n
    else:
        for file in os.listdir(path):
            curPath = os.path.join(path, file)
            if os.path.isdir(curPath):
                n = get_sum_words(curPath, n)
            else:
                vec = txt_to_dic(curPath)
                for freq in vec.values():
                    n += int(freq)
        return n