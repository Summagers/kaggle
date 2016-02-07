
# San Francisco Crime Classification
https://www.kaggle.com/c/sf-crime
## Predict the category of crimes that occurred in the city by the bay

From 1934 to 1963, San Francisco was infamous for housing some of the world's most notorious criminals on the inescapable island of Alcatraz.

Today, the city is known more for its tech scene than its criminal past. But, with rising wealth inequality, housing shortages, and a proliferation of expensive digital toys riding BART to work, there is no scarcity of crime in the city by the bay.

From Sunset to SOMA, and Marina to Excelsior, this competition's dataset provides nearly 12 years of crime reports from across all of San Francisco's neighborhoods. Given time and location, you must predict the category of crime that occurred.

# Basic Imports and Reads


```python
import numpy as np
import pandas
import sklearn

FILE_TRAIN = 'train.csv'
FILE_TEST  = 'test.csv'
with open(FILE_TRAIN, 'r') as f:
    dt = pandas.read_csv(f)
with open(FILE_TEST, 'r') as f:
    dt_test = pandas.read_csv(f)
```

# Exploration of Data
Here we do a basic exploration of the types of columns, number of rows, and the type of data they contain.


```python
dt
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dates</th>
      <th>Category</th>
      <th>DayOfWeek</th>
      <th>PdDistrict</th>
      <th>Address</th>
      <th>X</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-05-13 23:53:00</td>
      <td>WARRANTS</td>
      <td>Wednesday</td>
      <td>NORTHERN</td>
      <td>OAK ST / LAGUNA ST</td>
      <td>-122.425892</td>
      <td>37.774599</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-05-13 23:53:00</td>
      <td>OTHER OFFENSES</td>
      <td>Wednesday</td>
      <td>NORTHERN</td>
      <td>OAK ST / LAGUNA ST</td>
      <td>-122.425892</td>
      <td>37.774599</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-05-13 23:33:00</td>
      <td>OTHER OFFENSES</td>
      <td>Wednesday</td>
      <td>NORTHERN</td>
      <td>VANNESS AV / GREENWICH ST</td>
      <td>-122.424363</td>
      <td>37.800414</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-05-13 23:30:00</td>
      <td>LARCENY/THEFT</td>
      <td>Wednesday</td>
      <td>NORTHERN</td>
      <td>1500 Block of LOMBARD ST</td>
      <td>-122.426995</td>
      <td>37.800873</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-05-13 23:30:00</td>
      <td>LARCENY/THEFT</td>
      <td>Wednesday</td>
      <td>PARK</td>
      <td>100 Block of BRODERICK ST</td>
      <td>-122.438738</td>
      <td>37.771541</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2015-05-13 23:30:00</td>
      <td>LARCENY/THEFT</td>
      <td>Wednesday</td>
      <td>INGLESIDE</td>
      <td>0 Block of TEDDY AV</td>
      <td>-122.403252</td>
      <td>37.713431</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2015-05-13 23:30:00</td>
      <td>VEHICLE THEFT</td>
      <td>Wednesday</td>
      <td>INGLESIDE</td>
      <td>AVALON AV / PERU AV</td>
      <td>-122.423327</td>
      <td>37.725138</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2015-05-13 23:30:00</td>
      <td>VEHICLE THEFT</td>
      <td>Wednesday</td>
      <td>BAYVIEW</td>
      <td>KIRKWOOD AV / DONAHUE ST</td>
      <td>-122.371274</td>
      <td>37.727564</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2015-05-13 23:00:00</td>
      <td>LARCENY/THEFT</td>
      <td>Wednesday</td>
      <td>RICHMOND</td>
      <td>600 Block of 47TH AV</td>
      <td>-122.508194</td>
      <td>37.776601</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2015-05-13 23:00:00</td>
      <td>LARCENY/THEFT</td>
      <td>Wednesday</td>
      <td>CENTRAL</td>
      <td>JEFFERSON ST / LEAVENWORTH ST</td>
      <td>-122.419088</td>
      <td>37.807802</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2015-05-13 22:58:00</td>
      <td>LARCENY/THEFT</td>
      <td>Wednesday</td>
      <td>CENTRAL</td>
      <td>JEFFERSON ST / LEAVENWORTH ST</td>
      <td>-122.419088</td>
      <td>37.807802</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2015-05-13 22:30:00</td>
      <td>OTHER OFFENSES</td>
      <td>Wednesday</td>
      <td>TARAVAL</td>
      <td>0 Block of ESCOLTA WY</td>
      <td>-122.487983</td>
      <td>37.737667</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2015-05-13 22:30:00</td>
      <td>VANDALISM</td>
      <td>Wednesday</td>
      <td>TENDERLOIN</td>
      <td>TURK ST / JONES ST</td>
      <td>-122.412414</td>
      <td>37.783004</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2015-05-13 22:06:00</td>
      <td>LARCENY/THEFT</td>
      <td>Wednesday</td>
      <td>NORTHERN</td>
      <td>FILLMORE ST / GEARY BL</td>
      <td>-122.432915</td>
      <td>37.784353</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2015-05-13 22:00:00</td>
      <td>NON-CRIMINAL</td>
      <td>Wednesday</td>
      <td>BAYVIEW</td>
      <td>200 Block of WILLIAMS AV</td>
      <td>-122.397744</td>
      <td>37.729935</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2015-05-13 22:00:00</td>
      <td>NON-CRIMINAL</td>
      <td>Wednesday</td>
      <td>BAYVIEW</td>
      <td>0 Block of MENDELL ST</td>
      <td>-122.383692</td>
      <td>37.743189</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2015-05-13 22:00:00</td>
      <td>ROBBERY</td>
      <td>Wednesday</td>
      <td>TENDERLOIN</td>
      <td>EDDY ST / JONES ST</td>
      <td>-122.412597</td>
      <td>37.783932</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2015-05-13 21:55:00</td>
      <td>ASSAULT</td>
      <td>Wednesday</td>
      <td>INGLESIDE</td>
      <td>GODEUS ST / MISSION ST</td>
      <td>-122.421682</td>
      <td>37.742822</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2015-05-13 21:40:00</td>
      <td>OTHER OFFENSES</td>
      <td>Wednesday</td>
      <td>BAYVIEW</td>
      <td>MENDELL ST / HUDSON AV</td>
      <td>-122.386401</td>
      <td>37.738983</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2015-05-13 21:30:00</td>
      <td>NON-CRIMINAL</td>
      <td>Wednesday</td>
      <td>TENDERLOIN</td>
      <td>100 Block of JONES ST</td>
      <td>-122.412250</td>
      <td>37.782556</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2015-05-13 21:30:00</td>
      <td>LARCENY/THEFT</td>
      <td>Wednesday</td>
      <td>INGLESIDE</td>
      <td>200 Block of EVELYN WY</td>
      <td>-122.449389</td>
      <td>37.742669</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2015-05-13 21:17:00</td>
      <td>ROBBERY</td>
      <td>Wednesday</td>
      <td>INGLESIDE</td>
      <td>1600 Block of VALENCIA ST</td>
      <td>-122.420272</td>
      <td>37.747332</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2015-05-13 21:11:00</td>
      <td>WARRANTS</td>
      <td>Wednesday</td>
      <td>TENDERLOIN</td>
      <td>100 Block of JONES ST</td>
      <td>-122.412250</td>
      <td>37.782556</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2015-05-13 21:11:00</td>
      <td>NON-CRIMINAL</td>
      <td>Wednesday</td>
      <td>TENDERLOIN</td>
      <td>100 Block of JONES ST</td>
      <td>-122.412250</td>
      <td>37.782556</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2015-05-13 21:10:00</td>
      <td>LARCENY/THEFT</td>
      <td>Wednesday</td>
      <td>NORTHERN</td>
      <td>FILLMORE ST / LOMBARD ST</td>
      <td>-122.436049</td>
      <td>37.799841</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2015-05-13 21:00:00</td>
      <td>NON-CRIMINAL</td>
      <td>Wednesday</td>
      <td>TENDERLOIN</td>
      <td>300 Block of OFARRELL ST</td>
      <td>-122.410509</td>
      <td>37.786043</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2015-05-13 21:00:00</td>
      <td>LARCENY/THEFT</td>
      <td>Wednesday</td>
      <td>NORTHERN</td>
      <td>2000 Block of BUSH ST</td>
      <td>-122.431018</td>
      <td>37.787388</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2015-05-13 21:00:00</td>
      <td>LARCENY/THEFT</td>
      <td>Wednesday</td>
      <td>INGLESIDE</td>
      <td>500 Block of COLLEGE AV</td>
      <td>-122.423656</td>
      <td>37.732556</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2015-05-13 21:00:00</td>
      <td>LARCENY/THEFT</td>
      <td>Wednesday</td>
      <td>TARAVAL</td>
      <td>19TH AV / SANTIAGO ST</td>
      <td>-122.475773</td>
      <td>37.744919</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2015-05-13 20:56:00</td>
      <td>OTHER OFFENSES</td>
      <td>Wednesday</td>
      <td>TARAVAL</td>
      <td>2000 Block of 41ST AV</td>
      <td>-122.499787</td>
      <td>37.748518</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>878019</th>
      <td>2003-01-06 02:37:00</td>
      <td>OTHER OFFENSES</td>
      <td>Monday</td>
      <td>SOUTHERN</td>
      <td>6TH ST / MARKET ST</td>
      <td>-122.410294</td>
      <td>37.782231</td>
    </tr>
    <tr>
      <th>878020</th>
      <td>2003-01-06 02:32:00</td>
      <td>OTHER OFFENSES</td>
      <td>Monday</td>
      <td>NORTHERN</td>
      <td>VAN NESS AV / TURK ST</td>
      <td>-122.420642</td>
      <td>37.781961</td>
    </tr>
    <tr>
      <th>878021</th>
      <td>2003-01-06 02:24:00</td>
      <td>VANDALISM</td>
      <td>Monday</td>
      <td>NORTHERN</td>
      <td>SANCHEZ ST / 14TH ST</td>
      <td>-122.431191</td>
      <td>37.767595</td>
    </tr>
    <tr>
      <th>878022</th>
      <td>2003-01-06 02:16:00</td>
      <td>VEHICLE THEFT</td>
      <td>Monday</td>
      <td>MISSION</td>
      <td>17TH ST / MISSION ST</td>
      <td>-122.419516</td>
      <td>37.763429</td>
    </tr>
    <tr>
      <th>878023</th>
      <td>2003-01-06 02:15:00</td>
      <td>LARCENY/THEFT</td>
      <td>Monday</td>
      <td>TENDERLOIN</td>
      <td>600 Block of ELLIS ST</td>
      <td>-122.416894</td>
      <td>37.784286</td>
    </tr>
    <tr>
      <th>878024</th>
      <td>2003-01-06 02:09:00</td>
      <td>OTHER OFFENSES</td>
      <td>Monday</td>
      <td>PARK</td>
      <td>600 Block of DIVISADERO ST</td>
      <td>-122.437781</td>
      <td>37.775483</td>
    </tr>
    <tr>
      <th>878025</th>
      <td>2003-01-06 02:06:00</td>
      <td>OTHER OFFENSES</td>
      <td>Monday</td>
      <td>BAYVIEW</td>
      <td>NEWHALL ST / GALVEZ AV</td>
      <td>-122.387710</td>
      <td>37.740674</td>
    </tr>
    <tr>
      <th>878026</th>
      <td>2003-01-06 02:06:00</td>
      <td>WARRANTS</td>
      <td>Monday</td>
      <td>BAYVIEW</td>
      <td>NEWHALL ST / GALVEZ AV</td>
      <td>-122.387710</td>
      <td>37.740674</td>
    </tr>
    <tr>
      <th>878027</th>
      <td>2003-01-06 02:00:00</td>
      <td>WARRANTS</td>
      <td>Monday</td>
      <td>SOUTHERN</td>
      <td>900 Block of MARKET ST</td>
      <td>-122.409708</td>
      <td>37.782828</td>
    </tr>
    <tr>
      <th>878028</th>
      <td>2003-01-06 02:00:00</td>
      <td>ASSAULT</td>
      <td>Monday</td>
      <td>SOUTHERN</td>
      <td>6TH ST / MARKET ST</td>
      <td>-122.410294</td>
      <td>37.782231</td>
    </tr>
    <tr>
      <th>878029</th>
      <td>2003-01-06 01:54:00</td>
      <td>OTHER OFFENSES</td>
      <td>Monday</td>
      <td>TENDERLOIN</td>
      <td>1400 Block of GOLDEN GATE AV</td>
      <td>-122.434423</td>
      <td>37.779193</td>
    </tr>
    <tr>
      <th>878030</th>
      <td>2003-01-06 01:54:00</td>
      <td>SEX OFFENSES FORCIBLE</td>
      <td>Monday</td>
      <td>TENDERLOIN</td>
      <td>1400 Block of GOLDEN GATE AV</td>
      <td>-122.434423</td>
      <td>37.779193</td>
    </tr>
    <tr>
      <th>878031</th>
      <td>2003-01-06 01:50:00</td>
      <td>ASSAULT</td>
      <td>Monday</td>
      <td>BAYVIEW</td>
      <td>3RD ST / NEWCOMB AV</td>
      <td>-122.390417</td>
      <td>37.735593</td>
    </tr>
    <tr>
      <th>878032</th>
      <td>2003-01-06 01:36:00</td>
      <td>OTHER OFFENSES</td>
      <td>Monday</td>
      <td>NORTHERN</td>
      <td>GEARY BL / FRANKLIN ST</td>
      <td>-122.423031</td>
      <td>37.785482</td>
    </tr>
    <tr>
      <th>878033</th>
      <td>2003-01-06 01:30:00</td>
      <td>VANDALISM</td>
      <td>Monday</td>
      <td>RICHMOND</td>
      <td>1000 Block of 22ND AV</td>
      <td>-122.391668</td>
      <td>37.757793</td>
    </tr>
    <tr>
      <th>878034</th>
      <td>2003-01-06 01:30:00</td>
      <td>TRESPASS</td>
      <td>Monday</td>
      <td>RICHMOND</td>
      <td>1000 Block of 22ND AV</td>
      <td>-122.391668</td>
      <td>37.757793</td>
    </tr>
    <tr>
      <th>878035</th>
      <td>2003-01-06 00:55:00</td>
      <td>ASSAULT</td>
      <td>Monday</td>
      <td>NORTHERN</td>
      <td>1300 Block of WEBSTER ST</td>
      <td>-122.431046</td>
      <td>37.783030</td>
    </tr>
    <tr>
      <th>878036</th>
      <td>2003-01-06 00:55:00</td>
      <td>LARCENY/THEFT</td>
      <td>Monday</td>
      <td>NORTHERN</td>
      <td>1300 Block of WEBSTER ST</td>
      <td>-122.431046</td>
      <td>37.783030</td>
    </tr>
    <tr>
      <th>878037</th>
      <td>2003-01-06 00:55:00</td>
      <td>VANDALISM</td>
      <td>Monday</td>
      <td>NORTHERN</td>
      <td>1300 Block of WEBSTER ST</td>
      <td>-122.431046</td>
      <td>37.783030</td>
    </tr>
    <tr>
      <th>878038</th>
      <td>2003-01-06 00:42:00</td>
      <td>WARRANTS</td>
      <td>Monday</td>
      <td>TENDERLOIN</td>
      <td>TAYLOR ST / GEARY ST</td>
      <td>-122.411519</td>
      <td>37.786941</td>
    </tr>
    <tr>
      <th>878039</th>
      <td>2003-01-06 00:40:00</td>
      <td>OTHER OFFENSES</td>
      <td>Monday</td>
      <td>NORTHERN</td>
      <td>POLK ST / CALIFORNIA ST</td>
      <td>-122.420692</td>
      <td>37.790577</td>
    </tr>
    <tr>
      <th>878040</th>
      <td>2003-01-06 00:33:00</td>
      <td>ASSAULT</td>
      <td>Monday</td>
      <td>MISSION</td>
      <td>2800 Block of FOLSOM ST</td>
      <td>-122.414073</td>
      <td>37.751685</td>
    </tr>
    <tr>
      <th>878041</th>
      <td>2003-01-06 00:31:00</td>
      <td>OTHER OFFENSES</td>
      <td>Monday</td>
      <td>RICHMOND</td>
      <td>CLEMENT ST / 14TH AV</td>
      <td>-122.472985</td>
      <td>37.782552</td>
    </tr>
    <tr>
      <th>878042</th>
      <td>2003-01-06 00:20:00</td>
      <td>ASSAULT</td>
      <td>Monday</td>
      <td>BAYVIEW</td>
      <td>1500 Block of SHAFTER AV</td>
      <td>-122.389769</td>
      <td>37.730564</td>
    </tr>
    <tr>
      <th>878043</th>
      <td>2003-01-06 00:20:00</td>
      <td>OTHER OFFENSES</td>
      <td>Monday</td>
      <td>BAYVIEW</td>
      <td>1500 Block of SHAFTER AV</td>
      <td>-122.389769</td>
      <td>37.730564</td>
    </tr>
    <tr>
      <th>878044</th>
      <td>2003-01-06 00:15:00</td>
      <td>ROBBERY</td>
      <td>Monday</td>
      <td>TARAVAL</td>
      <td>FARALLONES ST / CAPITOL AV</td>
      <td>-122.459033</td>
      <td>37.714056</td>
    </tr>
    <tr>
      <th>878045</th>
      <td>2003-01-06 00:01:00</td>
      <td>LARCENY/THEFT</td>
      <td>Monday</td>
      <td>INGLESIDE</td>
      <td>600 Block of EDNA ST</td>
      <td>-122.447364</td>
      <td>37.731948</td>
    </tr>
    <tr>
      <th>878046</th>
      <td>2003-01-06 00:01:00</td>
      <td>LARCENY/THEFT</td>
      <td>Monday</td>
      <td>SOUTHERN</td>
      <td>5TH ST / FOLSOM ST</td>
      <td>-122.403390</td>
      <td>37.780266</td>
    </tr>
    <tr>
      <th>878047</th>
      <td>2003-01-06 00:01:00</td>
      <td>VANDALISM</td>
      <td>Monday</td>
      <td>SOUTHERN</td>
      <td>TOWNSEND ST / 2ND ST</td>
      <td>-122.390531</td>
      <td>37.780607</td>
    </tr>
    <tr>
      <th>878048</th>
      <td>2003-01-06 00:01:00</td>
      <td>FORGERY/COUNTERFEITING</td>
      <td>Monday</td>
      <td>BAYVIEW</td>
      <td>1800 Block of NEWCOMB AV</td>
      <td>-122.394926</td>
      <td>37.738212</td>
    </tr>
  </tbody>
</table>
<p>878049 rows × 7 columns</p>
</div>




```python
# Dataframe Info
dt.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 878049 entries, 0 to 878048
    Data columns (total 7 columns):
    Dates         878049 non-null object
    Category      878049 non-null object
    DayOfWeek     878049 non-null object
    PdDistrict    878049 non-null object
    Address       878049 non-null object
    X             878049 non-null float64
    Y             878049 non-null float64
    dtypes: float64(2), object(5)
    memory usage: 53.6+ MB



```python
# Types of Crimes
print dt.Category.nunique()  # Number of unique categories
dt.groupby('Category').size().sort_values(ascending=False)
```

    39





    Category
    LARCENY/THEFT                  174900
    OTHER OFFENSES                 126182
    NON-CRIMINAL                    92304
    ASSAULT                         76876
    DRUG/NARCOTIC                   53971
    VEHICLE THEFT                   53781
    VANDALISM                       44725
    WARRANTS                        42214
    BURGLARY                        36755
    SUSPICIOUS OCC                  31414
    MISSING PERSON                  25989
    ROBBERY                         23000
    FRAUD                           16679
    FORGERY/COUNTERFEITING          10609
    SECONDARY CODES                  9985
    WEAPON LAWS                      8555
    PROSTITUTION                     7484
    TRESPASS                         7326
    STOLEN PROPERTY                  4540
    SEX OFFENSES FORCIBLE            4388
    DISORDERLY CONDUCT               4320
    DRUNKENNESS                      4280
    RECOVERED VEHICLE                3138
    KIDNAPPING                       2341
    DRIVING UNDER THE INFLUENCE      2268
    RUNAWAY                          1946
    LIQUOR LAWS                      1903
    ARSON                            1513
    LOITERING                        1225
    EMBEZZLEMENT                     1166
    SUICIDE                           508
    FAMILY OFFENSES                   491
    BAD CHECKS                        406
    BRIBERY                           289
    EXTORTION                         256
    SEX OFFENSES NON FORCIBLE         148
    GAMBLING                          146
    PORNOGRAPHY/OBSCENE MAT            22
    TREA                                6
    dtype: int64




```python
# Convert Categories into numerical classes
cat_uniques = dt.Category.unique()
cat_to_num = {k: v for (k, v) in zip(cat_uniques, range(1, len(cat_uniques) + 1))}
num_to_cat = {k: v for (k, v) in zip(range(1, len(cat_uniques) + 1), cat_uniques)}
dt['CatClass'] = dt['Category']
dt['CatClass'] = dt['CatClass'].map(cat_class).astype(int)
```


```python
def convert(dt):
    return np.array([dt.X.values, dt.Y.values, dt.CatClass.values]).T
```


```python
# convert into model
xy = np.array([dt.X.values, dt.Y.values, dt.CatClass.values]).T
```

    [[-122.42589168   37.7745986     1.        ]
     [-122.42589168   37.7745986     2.        ]
     [-122.42436302   37.80041432    2.        ]
     ..., 
     [-122.40339036   37.78026558    3.        ]
     [-122.3905314    37.78060708    5.        ]
     [-122.39492572   37.73821154   13.        ]]



```python
# cut out validation set
from sklearn import cross_validation
X_train, X_valid, y_train, y_valid = cross_validation.train_test_split(xy[0::, 0:2], xy[0::, 2], test_size=0.4, random_state=0)
```


```python
# train the model
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(X_train, y_train)
```


```python
# score the results
forest.score(X_valid, y_valid)
```




    0.27537440920220946




```python
X_test = np.array([dt_test.X.values, dt_test.Y.values]).T
y_test = forest.predict(X_test)
```


```python
print y_test
```

    [  4.   2.  10. ...,   8.   3.   3.]



```python
headers = 'Id,' + ','.join(sorted(cat_uniques)) + '\n'
f = open('y_test.txt', 'w')
f.write(headers)
for i in xrange(len(y_test)):
    arr = [0] * 39
    arr[int(y_test[i])] = 1
    f.write('%s,%s\n' % (i, ','.join(map(str, arr))))
f.close()
```


```python

```
