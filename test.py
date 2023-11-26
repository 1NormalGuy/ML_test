from pre_processing import normalization as nm
import perceptron
import knn
import bayes
import svm
import os

def run_test(modelChoice, dataChoice):
    if modelChoice == 'perceptron':
        perceptron.perceptron_test(dataChoice=dataChoice, categories=['alt.atheism','comp.graphics'], max_iter=9, a=1)
    elif modelChoice == 'bayes':
        bayes.naive_bayes_test(dataChoice=dataChoice, lamb=0.2)
    elif modelChoice == 'knn':
        knn.knn_test(dataChoice=dataChoice, p=2, k=1)
    elif modelChoice == 'svm':
        svm.svm_test(category=['alt.atheism', 'comp.graphics', 'misc.forsale', 'rec.autos','sci.crypt', 'soc.religion.christian', 'talk.politics.guns'], C=1, kernel='linear')
    else:
        print("Invalid model choice. Please choose from 'perceptron', 'naive_bayes', 'knn', or 'svm'.")

run_test('svm', 'dataof2')