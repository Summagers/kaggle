import numpy as np
import pandas as pd
import random
import tensorflow as tf

from sklearn import datasets, cross_validation, metrics
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.utils import np_utils

import csv
from copy import copy


def make_predictors(data):
    for col in ['DayOfWeek','PdDistrict']:
        dummies = pd.get_dummies(data[col])
        data[col[0:3]+"_"+dummies.columns] = dummies

    data['PandasDates'] = pd.to_datetime(data['Dates'])
    data[['X','Y']] = preprocessing.normalize(data[['X','Y']], norm='l2')
    data['Year'] = data['PandasDates'].dt.year
    data['Month'] = data['PandasDates'].dt.month
    data['Day'] = data['PandasDates'].dt.dayofyear
    data['Hour'] = data['PandasDates'].dt.hour
    data['Minute'] = data['PandasDates'].dt.minute

    return data

def make_PCA(X, n_comp):
    pca = PCA(n_components=n_comp)
    pca.fit(X)
    return pca

def build_model(input_dim, output_dim, hn=32, dp=0.5, layers=1,
                init_mode='glorot_uniform',
                batch_norm=True):
    model = Sequential()
    model.add(Dense(hn, input_dim=input_dim, init=init_mode))
    model.add(Activation('relu'))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dp))

    for i in xrange(layers):
        model.add(Dense(hn, init=init_mode))
        model.add(Activation('relu'))
        if batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(dp))

    model.add(Dense(output_dim, init=init_mode))
    model.add(Activation('softmax'))

    return model


def save_model_weights(model, name):
    try:
        model.save_weights(name, overwrite=True)
    except:
        print "failed to save classifier weights"
    pass

def load_model_weights(model, name):
    try:
        model.load_weights(name)
    except:
        print "Can't load weights!"


def run_model(model,batch_size, nb_epoch, lr, load_name='SF-crime.h5', save_name='SF-crime.h5'):
    adam = Adam(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=adam)
    load_model_weights(model, load_name)
    model.fit(X_train,
              y_train_OH,
              nb_epoch=nb_epoch,
              batch_size=batch_size,
              validation_split=0.1,
              show_accuracy=True,
              verbose=True)

    save_model_weights(model, save_name)
    return model

use_PCA = True
save_preds = False 

data = make_predictors(pd.read_csv('train.csv'))
test_data = make_predictors(pd.read_csv('test.csv'))

train_cols = [col for col in data.columns if col not in ['DayOfWeek','PandasDates', 'PdDistrict','Category','Address','Dates','Descript','Resolution']]
X = data[train_cols]

y = data['Category'].astype('category').cat.codes

X = X.as_matrix()
if use_PCA:
    pca = make_PCA(X, 15)
    X = pca.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=42)

y_OH = np_utils.to_categorical(y.as_matrix(), y.nunique())
y_train_OH = np_utils.to_categorical(y_train.as_matrix(), y.nunique())
y_test_OH = np_utils.to_categorical(y_test.as_matrix(), y.nunique())

input_dim = X.shape[1]
output_dim = y_OH.shape[1]

model = build_model(input_dim, output_dim, hn=256, dp=0.5, layers=5, init_mode='glorot_normal')

model = run_model(model, 256, 1, 1e-2, load_name='SF-crime_FC256x5_PCA-15_train-0.5.h5', save_name='SF-crime_FC256x5_PCA-15_train-0.5.h5')

if save_preds:
    X_final_test = test_data[train_cols].as_matrix()
    X_final_test = pca.transform(X_final_test)
    pred = model.predict_proba(X_final_test, batch_size=256, verbose=1)

    labels = list(pd.get_dummies(data['Category']).columns)

    with open('sf-nn.csv', 'w') as outf:
        fo = csv.writer(outf, lineterminator='\n')
        fo.writerow(['Id'] + labels)
        for i, p in enumerate(pred):
            fo.writerow([i] + list(p))
