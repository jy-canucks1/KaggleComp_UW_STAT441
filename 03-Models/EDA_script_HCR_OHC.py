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

train_x_raw = pd.read_csv("X_train.csv", low_memory = False, index_col=0)
train_y_raw = pd.read_csv("y_train.csv", low_memory = False, index_col=0)
test_x_raw = pd.read_csv("X_test.csv", low_memory= False, index_col=0)

df_train = pd.DataFrame(train_x_raw)
df_test = pd.DataFrame(test_x_raw)
df_train_c = df_train.copy()
merged=pd.concat([df_train_c, df_test], ignore_index=True)

df_y = pd.DataFrame(train_y_raw)

############################################################# FUNCTIONS ###############################################################

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

from collections import Counter

def merge_columns(dat, colname):
    for name in colname:
        name_org = name.replace("_11c", "")
        dat.loc[dat[name_org] == -4, name_org] = dat.loc[dat[name_org] == -4, name]


def print_diff(varname):
  print(set(merged[varname].unique()).difference(set(df_test[varname].unique())))

def cumulatively_categorise(column,threshold=0.80,return_categories_list=True):
  #Find the threshold value using the percentage and number of instances in the column
  threshold_value=int(threshold*len(column))
  #Initialise an empty list for our new minimised categories 
  categories_list=[]
  #Initialise a variable to calculate the sum of frequencies
  s=0
  #Create a counter dictionary of the form unique_value: frequency
  counts=Counter(column)

  #Loop through the category name and its corresponding frequency after sorting the categories by descending order of frequency
  for i,j in counts.most_common():
    #Add the frequency to the global sum
    s+=dict(counts)[i]
    #Append the category name to the list
    categories_list.append(i)
    #Check if the global sum has reached the threshold value, if so break the loop
    if s>=threshold_value:
      break
  #Append the category Other to the list
  categories_list.append(-100)

  #Replace all instances not in our new categories by Other  
  new_column=column.apply(lambda x: x if x in categories_list else -100)

  #Return transformed column and unique values if return_categories=True
  if(return_categories_list):
    return new_column,categories_list
  #Return only the transformed column if return_categories=False
  else:
    return new_column
  
def simpleAggregation_helper(var, threshold):
  train=merged[var]
  test=df_test[var]
  cat = [train, test]
  df = pd.concat(cat)
  transformed_column=cumulatively_categorise(df, threshold, return_categories_list=False)
  tc_train=transformed_column[0:len(train)]
  tc_test=transformed_column[len(train):len(train)+len(test)]
  merged[var]=tc_train
  df_test[var]=tc_test

  
def simpleAggregation(variable, threshold=0.8):
    if isinstance(variable, str):
      simpleAggregation_helper(variable, threshold)
    elif isinstance(variable, list):
      for varname in variable:
        simpleAggregation_helper(varname, threshold)
        
### Convert fw_start ==> Start month of fw
### Convert fw_end ==> Duration of fw
def timeEDA(data):
    fw_start = data['fw_start']
    fw_end = data['fw_end']
    fieldwork_start_month = []
    fw_duration = []
    for obs in range(0, len(fw_end)):
        fw_start_year = int(str(fw_start[obs])[0:4])
        fw_start_month = int(str(fw_start[obs])[4:6])
        fw_end_year = int(str(fw_end[obs])[0:4])
        fw_end_month = int(str(fw_end[obs])[4:6])
        duration_year = fw_end_year - fw_start_year
        duration_month = fw_end_month - fw_start_month
        duration = 12*duration_year + duration_month
        fieldwork_start_month.append(fw_start_month)
        fw_duration.append(duration)
    data['fw_start'] = fieldwork_start_month
    data['fw_end'] = fw_duration
    data.rename(columns={'fw_start':'fw_start_month', 'fw_end':'fw_duration'}, inplace=True)



################################################################### GLOBAL VAR ###########################################################
columns_to_encode = []




##################################################### INITIAL DROP / MISSING DATA PREPROCESSING #################################################

columns_to_drop = ['c_abrv', 'f46_IT', 'v72_DE', 'v73_DE', 'v74_DE', 'v75_DE', 'v76_DE', 'v77_DE', 'v78_DE', 'v79_DE']
columns_to_drop += ['f252_edulvlb_CH']
merged.drop(columns=columns_to_drop, inplace=True)
df_test.drop(columns=columns_to_drop, inplace=True)

#columns_to_drop = ['v228b', 'v231b', 'v233b', 'v251b', 'f252_edulvlb_CH', 'v275b_N1', 'v275b_N2', 'v275c_N2', 'v281a']

# Imputation 
merged.fillna({'v231b_r': -3}, inplace=True)
df_test.fillna({'v231b_r': -3}, inplace=True)

merged.fillna({'v233b_r': -3}, inplace=True)
df_test.fillna({'v233b_r': -3}, inplace=True)

merged.fillna({'v251b_r': -3}, inplace=True)
df_test.fillna({'v251b_r': -3}, inplace=True)

merged.fillna({'v228b_r': -3}, inplace=True)
df_test.fillna({'v228b_r': -3}, inplace=True)


####################################################### Age-related variables processing #############################################
# v226 : respondent age year
# age age:respondent
# age_r age recorded (6 intervals)
# age_r2 age recoded (3 intervals)
# age_r3 age recoded (7 intervals)
## Keep age_r3
ages_to_drop = ['v226', 'age', 'age_r', 'age_r2']
merged.drop(columns=ages_to_drop, inplace=True)
df_test.drop(columns=ages_to_drop, inplace=True)
# DECIDE WHICH ONE TO KEEP AFTER EVALUATING 

############################################################################### HOUSEHOLD / SPOUSE ######################################################################

#################################### Education level-related variables drop ####################################
# v243*: educational level respondent: ... with variations
# keep v243_ISCED_3: educational level respondent: ISCED-code three digit  
v243_to_drop = ['v243_edulvlb', 'v243_edulvlb_2', 'v243_edulvlb_1', 'v243_ISCED_2', 'v243_ISCED_2b','v243_ISCED_1', 'v243_EISCED', 
                'v243_ISCED97', 'v243_8cat', 'v243_r', 'v243_cs', 'v243_cs_DE1', 'v243_cs_DE2', 'v243_cs_DE3', 'v243_cs_GB1', 'v243_cs_GB2']
merged.drop(columns=v243_to_drop, inplace=True)
df_test.drop(columns=v243_to_drop, inplace=True)

# ### Job kinds-related variables drop
### keep v246_ESeC : kind of job respondent
v246_to_drop = ['v246_ISCO_2', 'v246_SIOPS', 'v246_ISEI', 'v246_egp']
merged.drop(columns=v246_to_drop, inplace=True)
df_test.drop(columns=v246_to_drop, inplace=True)


# ### Partner Education Level variables drop
# keep v252_cs : educational level spouse/partner:
v252_to_drop = ['v252_edulvlb', 'v252_edulvlb_1', 'v252_ISCED_3', 'v252_ISCED_2', 'v252_ISCED_2b', 'v252_ISCED_1', 'v252_EISCED', 'v252_ISCED97', 
                'v252_8cat', 'v252_r', 'v252_edulvlb_2', 'v252_cs_DE1', 'v252_cs_DE2', 'v252_cs_DE3', 'v252_cs_GB1', 'v252_cs_GB2']
merged.drop(columns=v252_to_drop, inplace=True)
df_test.drop(columns=v252_to_drop, inplace=True)


# ### Kind of job partner variables drop
# keep v255_ESeC: kind of job spouse/partner
v255_to_drop = ['v255_ISCO_2', 'v255_SIOPS', 'v255_ISEI', 'v255_egp']
merged.drop(columns=v255_to_drop, inplace=True)
df_test.drop(columns=v255_to_drop, inplace=True)


################################################### Households income variables to drop ################################################
merged.drop('v261_ppp', inplace=True, axis=1)
df_test.drop('v261_ppp', inplace=True, axis=1)


################################################## education level father/mother variables drop ################################################

# keep v262_cs: educational level father: ESS-edulvlb coding two digits 
v262_to_drop = ['v262_edulvlb', 'v262_edulvlb_1', 'v262_ISCED_3', 'v262_ISCED_2', 'v262_ISCED_2b', 'v262_ISCED_1', 'v262_EISCED', 'v262_ISCED97', 
                'v262_8cat', 'v262_r', 'v262_edulvlb_2', 'v262_cs_DE1', 'v262_cs_DE2', 'v262_cs_DE3', 'v262_cs_GB1', 'v262_cs_GB2']
merged.drop(columns=v262_to_drop, inplace=True)
df_test.drop(columns=v262_to_drop, inplace=True)

# keep v263_cs:educational level mother: ESS-edulvlb coding two digits
v263_to_drop = ['v263_edulvlb', 'v263_edulvlb_2', 'v263_edulvlb_1', 'v263_ISCED_3', 'v263_ISCED_2', 'v263_ISCED_2b', 'v263_ISCED_1', 'v263_EISCED',
                 'v263_ISCED97', 'v263_8cat', 'v263_r', 'v263_edulvlb_2', 'v263_cs_DE1', 'v263_cs_DE2', 'v263_cs_DE3', 'v263_cs_GB1', 'v263_cs_GB2']
merged.drop(columns=v263_to_drop, inplace=True)
df_test.drop(columns=v263_to_drop, inplace=True)


################################################### Interview dates variables drop ########################################################################
# v277: date of interview 
# v278a: time of interview: start hour 
# v278b: time of interview: start minute 
# v278c_r: time of interview: start  
# v279a: time of interview: end hour 
# v279b: time of interview: end minute 
# v279c_r: time of interview: end 
# v279d_r: time of interview: duration in minutes 

##### Keep v278a, v279d_r -- Duration in miniutes
times_to_drop = ['v277', 'v278b', 'v278c_r', 'v279a', 'v279b', 'v279c_r']
merged.drop(columns=times_to_drop, inplace=True)
df_test.drop(columns=times_to_drop, inplace=True)


############################################################### MERGE COLUMNS ##############################################################
merge_colname = find_colname_end(merged, '_11c')
merge_columns(merged, merge_colname)
merge_columns(df_test, merge_colname)

# print(find_colname(train_x_raw, 'c', 'endwith'))
# print(find_colname(train_x_raw, '_r', 'endwith'))
### Find variables containing _cs and do SimpleAggregation
# print(find_colname(df_train, '_cs', 'endwith'))
#aggregatecol = find_colname_end(df_train, '_cs')
#simpleAggregation(aggregatecol) #### TRAIN/TEST BOTH APPLICABLE


########################################################## TIME FIX ##################################################
### Convert fw_start ==> Start month of fw
### Convert fw_end ==> Duration of fw
timeEDA(merged)
timeEDA(df_test)

df_numerical_train = merged.copy()
for col in merged.columns:
  if pd.api.types.is_string_dtype(merged[col]):
    df_numerical_train.drop(col, inplace =True, axis=1)

corr = df_numerical_train.corr()
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
    dic[e] = df_numerical_train[e].corr(df_y['label'])

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

## List to be dropped (corr>0.95)
## ['v275c_N2', 'v275c_N1', 'v20a', 'v30b', 'v45b', 'v96a', 'v136_11c', 'v135_11c', 'v141_11c', 'v136_11c', 'v141_11c', 'v176_DK', 'v176_DK', 'v176_DK', 'v176_DK', 'v176_DK', 'v176_DK', 'v176_DK',
## 'v177_DK', 'v177_DK', 'v177_DK', 'v181_DK', 'v177_DK', 'v177_DK', 'v179_DK', 'v180_DK', 'v181_DK', 'v182_DK', 
## 'v183_DK', 'v180_DK', 'v181_DK', 'v182_DK', 'v183_DK', 'v181_DK', 'v180_DK', 'v183_DK', 'v181_DK', 'v181_DK', 'v183_DK', 'v222_DK', 'v223_DK', 'v224_DK', 'v222_DK', 'v224_DK', 'v224_DK', 'v275c_N2']

# List to be dropped (Too wide CI)
columns_to_drop = ['v24a_IT', 'v52', 'v54', 'v64', 'f96', 'v102', 'v129', 'v172', 'v184', 'v171', 'v215', 'v174_LR']
merged.drop(columns=columns_to_drop, inplace=True, axis=1)
df_test.drop(columns=columns_to_drop, inplace=True, axis=1)



##################################################### ONE HOT ENCODING ##################################################
columns_to_drop = ['v228b_r','v231b_r','v233b_r','v251b_r','v275c_N2', 'v275c_N1', 'v281a_r']
c1=set(columns_to_drop)
columns_to_drop = list(c1.union(set(dropped)))
merged.drop(columns=columns_to_drop, inplace=True)
df_test.drop(columns=columns_to_drop, inplace=True)

columns_to_encode = ['v228b', 'v231b', 'v233b', 'v251b', 'v275b_N1', 'v275b_N2', 'v281a']
columns_to_encode += find_colname_end(merged, '_cs')
columns_to_encode += ['v246_ESeC','v255_ESeC']

merged = pd.get_dummies(merged, columns=columns_to_encode)
df_test = pd.get_dummies(df_test, columns=columns_to_encode)
merged = merged.reindex(columns = sorted(merged.columns))
df_test = df_test.reindex(columns = sorted(df_test.columns))
################################################ 'DE' / 'GB' Country Specific Dropped ##################################################


## removed the column having 'GB'
merged.drop(list(merged.filter(regex='DE')), axis=1, inplace=True)
df_test.drop(list(df_test.filter(regex='DE')), axis=1, inplace=True)

## removed the column having 'GB'
merged.drop(list(merged.filter(regex='GB')), axis=1, inplace=True)
df_test.drop(list(df_test.filter(regex='GB')), axis=1, inplace=True)




###################################################################### CORRELATION CHECKUP ##########################################################################
df_train = merged.iloc[:len(df_train),:].copy()
df_test = merged.iloc[(len(df_train)-1):,:].copy()





# %%
