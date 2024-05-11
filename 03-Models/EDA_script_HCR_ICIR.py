import pandas as pd
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set options
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999

train_x_raw = pd.read_csv("X_train.csv", low_memory = True, index_col=0)
train_y_raw = pd.read_csv("y_train.csv", low_memory = True, index_col=0)
test_x_raw = pd.read_csv("X_test.csv", low_memory=True, index_col=0)

df_train = pd.DataFrame(train_x_raw)
df_test = pd.DataFrame(test_x_raw)
df_y = pd.DataFrame(train_y_raw)




### Variable 1 - 146 Preprocessing
columns_to_drop = ['c_abrv', 'f46_IT', 'v72_DE', 'v73_DE', 'v74_DE', 'v75_DE', 'v76_DE', 'v77_DE', 'v78_DE', 'v79_DE']
df_train.drop(columns=columns_to_drop, inplace=True)
df_test.drop(columns=columns_to_drop, inplace=True)

### Variable 147 - 292 Preprocessing
### Function to find the targeted colname
def find_colname_start(data, target):
  temp = []
  for varname in data.columns:
      if varname.startwith(target):
        temp.append(varname)
  return(temp)
  
def find_colname_end(data, target):
  temp = []
  for varname in data.columns:
      if varname.endswith(target):
        temp.append(varname)
  return(temp)

def merge_columns(dat, colname):
    for name in colname:
        name_org = name.replace("_11c", "")
        dat.loc[dat[name_org] == -4, name_org] = dat.loc[dat[name_org] == -4, name]



### Variable 293 - 438 Preprocessing
## removed string type data
df_train.drop('v228b', inplace=True, axis=1) 
df_test.drop('v228b', inplace=True, axis=1) 

df_train.fillna({'v228b_r': -3}, inplace=True)
df_test.fillna({'v228b_r': -3}, inplace=True)

df_train.drop('v231b', inplace=True, axis=1) 
df_test.drop('v231b', inplace=True, axis=1)

df_train.fillna({'v231b_r': -3}, inplace=True)
df_test.fillna({'v231b_r': -3}, inplace=True)

df_train.drop('v233b', inplace=True, axis=1)
df_test.drop('v233b', inplace=True, axis=1)

df_train.fillna({'v233b_r': -3}, inplace=True)
df_test.fillna({'v233b_r': -3}, inplace=True)

df_train.drop('v251b', inplace=True, axis=1)
df_test.drop('v251b', inplace=True, axis=1) 

df_train.fillna({'v251b_r': -3}, inplace=True)
df_test.fillna({'v251b_r': -3}, inplace=True)

df_train.drop('f252_edulvlb_CH', inplace=True, axis=1)
df_test.drop('f252_edulvlb_CH', inplace=True, axis=1)

## removed the column having 'DE'
df_train.drop(list(df_train.filter(regex='DE')), axis=1, inplace=True)
df_test.drop(list(df_test.filter(regex='DE')), axis=1, inplace=True)

## removed the column having 'GB'
df_train.drop(list(df_train.filter(regex='GB')), axis=1, inplace=True)
df_test.drop(list(df_test.filter(regex='GB')), axis=1, inplace=True)

df_train.drop('v281a', inplace=True, axis=1)
df_test.drop('v281a', inplace=True, axis=1)

label_mapping = {-1: 0, 1: 1, 2: 2, 3: 3, 4: 4}
df_y = df_y.replace(label_mapping)

df_train.drop('v275b_N2', inplace=True, axis=1)
df_test.drop('v275b_N2', inplace=True, axis=1)

df_train.drop('v275b_N1', inplace=True, axis=1)
df_test.drop('v275b_N1', inplace=True, axis=1) 

########################################### corr more than 0.95
corr = df_train.corr()
pairs = []

for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):  # i+1 to exclude self-correlation
        if (0.95 <= corr.iloc[i, j] <= 1) or (-1 <= corr.iloc[i, j] <= -0.95):
            pairs.append((corr.columns[i], corr.columns[j]))


set_pairs = []

for e in pairs:
     set_pairs.append(set(e))

x = list(set().union(*set_pairs))

dic = {}
for e in x:
    dic[e] = df_train[e].corr(df_y['label'])

remainder = []
for i in set_pairs:
    i = list(i)
    if abs(dic[i[0]]) > abs(dic[i[1]]):
        remainder.append(i[0])
    else:
        remainder.append(i[1])
dropped = []
for i in set_pairs:
    i = list(i)
    if abs(dic[i[0]]) < abs(dic[i[1]]):
        dropped.append(i[0])
    else:
        dropped.append(i[1])
for e in dropped:
    if not e in df_train.columns :
        continue
    df_train.drop(e, inplace=True, axis=1)
    df_test.drop(e, inplace=True, axis=1)

df_train.drop('v24a_IT', inplace=True, axis=1)
df_test.drop('v24a_IT', inplace=True, axis=1)

df_train.drop('v52', inplace=True, axis=1)
df_test.drop('v52', inplace=True, axis=1)

df_train.drop('v54', inplace=True, axis=1)
df_test.drop('v54', inplace=True, axis=1)

df_train.drop('v64', inplace=True, axis=1)
df_test.drop('v64', inplace=True, axis=1)

df_train.drop('f96', inplace=True, axis=1)
df_test.drop('f96', inplace=True, axis=1)

df_train.drop('v102', inplace=True, axis=1)
df_test.drop('v102', inplace=True, axis=1)

df_train.drop('v129', inplace=True, axis=1)
df_test.drop('v129', inplace=True, axis=1)

df_train.drop('v172', inplace=True, axis=1)
df_test.drop('v172', inplace=True, axis=1)

df_train.drop('v184', inplace=True, axis=1)
df_test.drop('v184', inplace=True, axis=1)

df_train.drop('v171', inplace=True, axis=1)
df_test.drop('v171', inplace=True, axis=1)


df_train.drop('v215', inplace=True, axis=1)
df_test.drop('v215', inplace=True, axis=1)

df_train.drop('v174_LR', inplace=True, axis=1)
df_test.drop('v174_LR', inplace=True, axis=1)

