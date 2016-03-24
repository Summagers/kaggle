#!/usr/bin/env python

import cPickle as pickle
import pandas as pd
from time import time

# scikit-learn
from sklearn import preprocessing
from sklearn.decomposition import PCA

# keras
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils

FILE_TRAIN  = 'train.csv'
FILE_TEST   = 'test.csv'
OUTPUT_FILE = 'y_test_nn.csv'

OLD_WEIGHT = 'y_test_nn_1024x2_4.h5'
NEW_WEIGHT = 'y_test_nn_1024x2_5.h5'

# Default parameters
N_EPOCHS = 20  # 15
N_LAYERS = 1
N_HN     = 1024
BATCH    = 250
DP       = 0.3
VALID    = 0.1
VERBOSE  = True

# Helper function to bucket month into seasons
def get_season(month):
  spring = summer = fall = winter = 0
  if 3 <= month <= 5:
    spring = 1
  elif 6 <= month <= 8:
    summer = 1
  elif 9 <= month <= 11:
    fall = 1
  else:
    winter = 1
  return spring, summer, fall, winter


def get_street_name(address):
  if '/' in address:
    address = map(str.strip, address.split('/'))
    return 0, ','.join(address)
  tokens = address.split(' Block of ')
  return int(tokens[0]), tokens[1]


# Main function to process whole data set and generate features
def process(df_orig, log_st=None):
  df = df_orig.copy()
  df['DateTimes'] = pd.to_datetime(df.Dates)
  df['DayOfWeek'] = df.DateTimes.dt.dayofweek
  df['DayOfYear'] = df.DateTimes.dt.dayofyear
  df['Year']      = df.DateTimes.dt.year
  df['Month']     = df.DateTimes.dt.month
  df['Hour']      = df.DateTimes.dt.hour
  df['Spring'], df['Summer'], df['Fall'], df['Winter'] = zip(*df.Month.apply(get_season))

  df['isWeekend']      = df.DayOfWeek.apply(lambda x: 1 if x in ('Saturday', 'Sunday') else 0)
  df['isAwake']        = df.Hour.apply(lambda x: 1 if (x == 0 or 8 <= x <= 23) else 0)
  df['isIntersection'] = df.Address.apply(lambda x: 1 if '/' in x else 0)

  df['Street'] = df.Address.apply(lambda x: x.strip())
  df['Block'], df['Streets'] = zip(*df.Street.apply(lambda x: get_street_name(x)))

  if log_st:
    df['log_street'] = df.Street.apply(lambda x: log_st.get(x, 0))

  districts = pd.get_dummies(df.PdDistrict, prefix='PD')
  df = pd.concat([df, districts], axis=1)
  for d in PDDISTRICTS:
    if 'PD_%s' % d not in df.columns:
      df[d] = 0

  cols = ['Id', 'Dates', 'DateTimes', 'PdDistrict', 'Address',  'Street', 'Streets', 'Category']
  df.drop(cols, axis=1, inplace=True, errors='ignore')

  return df


def nn(X, y, hn=None, n_layers=1, dp=0.5, epochs=1, init='glorot_uniform', valid=0, batch_size=64, verbose=False):
  # input and output layer size
  input_dim  = X.shape[1]
  output_dim = len(CATEGORIES)
  if not hn:
    hn = (input_dim + output_dim) / 2
  y = np_utils.to_categorical(y)

  # build neural network
  model = Sequential()
  model.add(Dense(hn, input_dim=input_dim, init=init))
  model.add(PReLU(input_shape=(hn,)))

  # add hidden layers
  for i in xrange(n_layers):
    model.add(Dense(hn, input_dim=hn, init=init))
    model.add(PReLU(input_shape=(hn,)))
    model.add(BatchNormalization((hn,)))
    model.add(Dropout(dp))

  model.add(Dense(output_dim, input_dim=hn, init=init))
  model.add(Activation('softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam')

  if OLD_WEIGHT:
    try:
      model.load_weights(OLD_WEIGHT)
      print 'Loaded weights: %s!' % OLD_WEIGHT
    except:
      print 'Error loading weights!'

  # validate on test set
  if valid > 0:
    fitting = model.fit(X, y, nb_epoch=epochs, batch_size=batch_size, shuffle=True,
                        show_accuracy=True, verbose=verbose, validation_split=valid)
  else:
    model.fit(X, y, nb_epochs=epochs, batch_size=batch_size, shuffle=True, verbose=verbose)
    fitting = score = 0
  return model, fitting


if __name__ == '__main__':

  tick = time()
  print 'Loading files...',
  with open(FILE_TRAIN, 'r') as f:
      dt = pd.read_csv(f)
  with open(FILE_TEST, 'r') as f:
      dt_test = pd.read_csv(f)
  with open('logit_street.pkl', 'r') as f:
      log_st = pickle.load(f)
  print '%.3f' % (time() - tick)

  CATEGORIES  = dt.Category.unique()
  PDDISTRICTS = dt.PdDistrict.unique()

  dt.drop(['Descript', 'Resolution'], axis=1, inplace=True, errors='ignore')
  dt['Category'] = dt.Category.astype('category')
  dt = dt.sample(frac=1)

  tick = time()
  print 'Eliminating bad coordinates...',
  avg_district = dt.groupby('PdDistrict')['X', 'Y'].mean()
  for pdd in PDDISTRICTS:
    dt.loc[(dt['Y']==90) & (dt['PdDistrict'] == pdd), 'X'] = avg_district['X'][pdd]
    dt.loc[(dt['Y']==90) & (dt['PdDistrict'] == pdd), 'Y'] = avg_district['Y'][pdd]
  print '%.3f' % (time() - tick)

  tick = time()
  print 'Processing...',
  ytrain = dt.Category.cat.rename_categories(range(len(CATEGORIES)))
  xtrain = process(dt, log_st)
  features = xtrain.columns.values
  print '%.3f' % (time() - tick)

  tick = time()
  print 'Feature scaling...',
  scaler = preprocessing.StandardScaler()
  scaler.fit(xtrain)
  xtrain[features] = scaler.transform(xtrain)
  print '%.3f' % (time() - tick)

  tick = time()
  print 'Feature reducing...',
  pca = PCA(n_components=15)
  pca.fit(xtrain)
  xtrain = pca.transform(xtrain)
  print '%.3f' % (time() - tick)

  tick = time()
  print 'Fitting the model...',
  model, fitting = nn(xtrain, ytrain, hn=N_HN, n_layers=N_LAYERS, dp=DP, epochs=N_EPOCHS,
                      batch_size=BATCH, valid=VALID, verbose=VERBOSE)
  print '%.3f' % (time() - tick)

  model.save_weights(NEW_WEIGHT)
  print 'Saved new weights: %s' % NEW_WEIGHT

  print 'Processing test dataset...',
  for pdd in PDDISTRICTS:
    dt_test.loc[(dt_test['Y'] == 90) & (dt_test['PdDistrict'] == pdd), 'X'] = avg_district['X'][pdd]
    dt_test.loc[(dt_test['Y'] == 90) & (dt_test['PdDistrict'] == pdd), 'Y'] = avg_district['Y'][pdd]

  xtest = process(dt_test, log_st)
  xtest[features] = scaler.transform(xtest)
  xtest = pca.transform(xtest)
  print '%.3f' % (time() - tick)

  print 'Predicting test set...',
  y_test = model.predict_proba(xtest)
  submission = pd.DataFrame(y_test, index=dt_test.index, columns=sorted(CATEGORIES))
  submission.to_csv(OUTPUT_FILE, index_label='Id')
  print '%.3f' % (time() - tick)
