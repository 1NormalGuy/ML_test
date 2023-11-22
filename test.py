from pre_processing import normalization as nm
import bayes
import os
import perceptron

# bayes.naive_bayes_test(dataChoice='dataof2', lamb=0.2)
perceptron.perceptron_test(dataChoice='dataof2', categories=['alt.atheism','comp.graphics'], max_iter=9, a=1)