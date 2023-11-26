from pre_processing import normalization as nm
import perceptron
import knn
import bayes
import svm
import os
# perceptron.perceptron_test(dataChoice='dataof2', categories=['alt.atheism','comp.graphics'], max_iter=8, a=1)
# bayes.naive_bayes_test(dataChoice='dataof20', lamb=0.2)
knn.knn_test(dataChoice='dataof20', p=2, k=1)
# svm.svm_test(category=['alt.atheism', 'comp.graphics', 'misc.forsale', 'rec.autos','sci.crypt', 'soc.religion.christian', 'talk.politics.guns'], C=1, kernel='linear')