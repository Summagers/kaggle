
# coding: utf-8

# # San Francisco Crime Classification
# https://www.kaggle.com/c/sf-crime
# ## Predict the category of crimes that occurred in the city by the bay
# 
# From 1934 to 1963, San Francisco was infamous for housing some of the world's most notorious criminals on the inescapable island of Alcatraz.
# 
# Today, the city is known more for its tech scene than its criminal past. But, with rising wealth inequality, housing shortages, and a proliferation of expensive digital toys riding BART to work, there is no scarcity of crime in the city by the bay.
# 
# From Sunset to SOMA, and Marina to Excelsior, this competition's dataset provides nearly 12 years of crime reports from across all of San Francisco's neighborhoods. Given time and location, you must predict the category of crime that occurred.

# # Basic Imports and Reads

# In[134]:

import numpy as np
import pandas
import sklearn

FILE_TRAIN = 'train.csv'
FILE_TEST  = 'test.csv'
with open(FILE_TRAIN, 'r') as f:
    dt = pandas.read_csv(f)
with open(FILE_TEST, 'r') as f:
    dt_test = pandas.read_csv(f)


# # Exploration of Data
# Here we do a basic exploration of the types of columns, number of rows, and the type of data they contain.

# In[56]:

dt


# In[57]:

# Dataframe Info
dt.info()


# In[61]:

# Types of Crimes
print dt.Category.nunique()  # Number of unique categories
dt.groupby('Category').size().sort_values(ascending=False)


# In[137]:

# Convert Categories into numerical classes
cat_uniques = dt.Category.unique()
cat_to_num = {k: v for (k, v) in zip(cat_uniques, range(1, len(cat_uniques) + 1))}
num_to_cat = {k: v for (k, v) in zip(range(1, len(cat_uniques) + 1), cat_uniques)}
dt['CatClass'] = dt['Category']
dt['CatClass'] = dt['CatClass'].map(cat_class).astype(int)


# In[132]:

def convert(dt):
    return np.array([dt.X.values, dt.Y.values, dt.CatClass.values]).T


# In[112]:

# convert into model
xy = np.array([dt.X.values, dt.Y.values, dt.CatClass.values]).T


# In[122]:

# cut out validation set
from sklearn import cross_validation
X_train, X_valid, y_train, y_valid = cross_validation.train_test_split(xy[0::, 0:2], xy[0::, 2], test_size=0.4, random_state=0)


# In[123]:

# train the model
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(X_train, y_train)


# In[124]:

# score the results
forest.score(X_valid, y_valid)


# In[135]:

X_test = np.array([dt_test.X.values, dt_test.Y.values]).T
y_test = forest.predict(X_test)


# In[148]:

print y_test


# In[157]:

headers = 'Id,' + ','.join(sorted(cat_uniques)) + '\n'
f = open('y_test.txt', 'w')
f.write(headers)
for i in xrange(len(y_test)):
    arr = [0] * 39
    arr[int(y_test[i])] = 1
    f.write('%s,%s\n' % (i, ','.join(map(str, arr))))
f.close()


# In[ ]:



