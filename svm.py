import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
import timeit

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# used to unwrap data
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# fine labels
# train data CF100
if False:
  data_train = unpickle('./train')
  x_train = data_train[b'data']
  y_train = np.array(data_train[b'fine_labels'])

# coarse labels
# train data CF100
if True:
  data_train = unpickle('./train')
  x_train = data_train[b'data']
  y_train = np.array(data_train[b'coarse_labels'])

print(x_train.shape)
print(y_train.shape)

# fine labels
if False:
  # test data CF100
  data_test= unpickle('./test')
  x_test= data_test[b'data']
  y_test= np.array(data_test[b'fine_labels'])

# coarse labels
if True:
  # test data CF100
  data_test= unpickle('./test')
  x_test= data_test[b'data']
  y_test= np.array(data_test[b'coarse_labels'])

print(x_test.shape)
print(y_test.shape)

# Preprocess names to take away file extensions
names_train_raw = data_train[b'filenames']

names_train = []

for x in names_train_raw:
  name_str = x.decode("utf-8")
  ic = name_str.find('_s_')
  name = name_str[:ic]
  names_train.append(name)

# make tuple of names to label
l_n_train = [(y_train[i], x) for i, x in enumerate(names_train)]

print(l_n_train)

# Preprocess names to take away file extensions
names_test_raw = data_test[b'filenames']

names_test = []

for x in names_test_raw:
  name_str = x.decode("utf-8")
  ic = name_str.find('_s_')
  name = name_str[:ic]
  names_test.append(name)

# make tuple of names to label
l_n_test = [(y_test[i], x) for i, x in enumerate(names_test)]

print(l_n_test)



t1 = timeit.default_timer()

clf = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
clf.fit(x_train[:10000], y_train[:10000])
t2 = timeit.default_timer()
print(t2-t1)
result = clf.score(x_test, y_test)
print('Result:', result)

t2 = timeit.default_timer()
print(t2-t1)
