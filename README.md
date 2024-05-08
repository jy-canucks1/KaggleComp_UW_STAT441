# KaggleComp_UW_STAT441

Link : [https://www.kaggle.com/competitions/w2024-kaggle-contest/overview](https://www.kaggle.com/competitions/w2024-kaggle-contest/overview)


## Overview

W2024 Kaggle Contest - Feb 2nd 2024 ~ Mar 29th 2024

### Description
___
In this kaggle contest, we will use a dataset to predict answers to the question "How important is religion in your life?". There are five possible responses for this question, which are: "no answer, very important, quite important, not important, not at all important".


Each observation represents a survey response of one person in different European countries. Your goal is to predict the correct response for this question based on the training dataset.


A more detailed description for each column could be found in the given file "codebook.txt".

### Evaluation
___
The evaluation metric for this competition is Multiclass Logarithmic Loss, which is the negative log likelihood divided by the number of observations. Lower is better.


![K-089](https://github.com/jy-canucks1/KaggleComp_UW_STAT441/assets/84373345/1bf6da96-c6e1-4aaf-a10f-39831085c373)

where:


N is the number of observations.

M is the number of classes.

y_{ij} is an indicator, being 1 if sample i belongs to class j, and 0 otherwise.

p_{ij} is the probability predicted by the model that sample i belongs to class j.


### Submission Format
___
Submission files should contain six columns: id, no answer, very important, quite important, not important, not at all important, where id column should contain the index id for the test data. Rest columns should represent the probabilities of being classified to the corresponding category. Your submission should have a header.

Submission Format example

```
id,no answer,very important,quite important,not important,not at all important
0,0.2,0.2,0.2,0.2,0.2
1,0.2,0.2,0.2,0.2,0.2
2,0.2,0.2,0.2,0.2,0.2
3,0.2,0.2,0.2,0.2,0.2
...
```

### Dataset Description
___
In this kaggle contest, we will use a dataset of survey results to predict "how is religion important in one's life". There are five possible responses for this question, which are: {'no answer':-1, 'very important':1, 'quite important':2, 'not important':3, 'not at all important':4}.


The data contains 48,000 observations for training data and 11438 observations for test data.


Each observation is a survey response of one person in different European countries.


Files

X_train.csv - the training set without target

X_test.csv - the test set without target

y_train.csv - the training set of target

sample_submission.csv - a sample submission file in the correct format

codebook.txt - (IMPORTANT) detailed descriptions of each column


### Software and Packages
___
#### Data Preprocessing

Language: Python, R

Packages: Pandas, Numpy (Python), Basic packages for previous STAT 441 assignments(R) 

#### Model

Language: Python

Model name: Xgboost, Neural Network, LightGBM, SVM, Random Forest

Packages: Numpy, Pandas, Scikit Learn, Xgboost, Pytorch, lightgbm, Bayesian Optimization

### Exploratory Data Analysis
___
![image](https://github.com/jy-canucks1/KaggleComp_UW_STAT441/assets/84373345/5eb599b7-83f9-41fc-8fcb-3956be4b7825)

Our dataset comprises 48,000 training and 11,438 test observations, each reflecting a survey response on the significance of religion in an individual's life across various European countries.

#### Data cleaning
* Columns with NA values → Imputation with ‘-3’

* Dropped country specific columns (*_GB, *_DE)

* Dropped non-numerical variables after confirming the relation with numerical type variables

#### Merge Column

* Merged “Modified” questionnaire columns with their original questionnaire columns. (ex: variables end of *_11c)

* Sum of v133_11c which is not -4 => 26+222+184+...+440 = 2985

![image](https://github.com/jy-canucks1/KaggleComp_UW_STAT441/assets/84373345/080c773c-df6a-4ef6-9378-8dc299b814c3)

#### Aggregation

* Some variables have extensive unique values (ex: v252_cs : 525 unique) => High Cardinality

* Used simple aggregation method to merge categories

* Aggregated until the sum of frequency is above 90%

#### Time Variable Adjusted

* Dropped irrelevant time data (ex: fw_start, fw_end, v278a, ...)

* Created new columns to compensate such dropped columns (ex:fw_duration, fw_start_month)

![image](https://github.com/jy-canucks1/KaggleComp_UW_STAT441/assets/84373345/f098384d-4106-4f12-bf60-f6de8a8e2fdb)

![image](https://github.com/jy-canucks1/KaggleComp_UW_STAT441/assets/84373345/edf09aa5-6d52-4d20-9dd7-385028f4d560)


#### Checking high correlation

* Dropped Variables with correlation value over 0.95 to lower overfitting

* Dropped features:
  'v275c_N1', 'v20a', 'v30b', 'v45b', 'v96a', 'v136_11c', 'v135_11c', 'v141_11c', 'v176_DK', 'v177_DK', 'v181_DK', 'v179_DK', 'v180_DK', 'v182_DK', 'v183_DK', 'v222_DK', 'v223_DK', 'v224_DK',      'age_r', 'age_r3'

![image](https://github.com/jy-canucks1/KaggleComp_UW_STAT441/assets/84373345/ef884fc9-1624-4046-9454-0b964afdb482)

#### Checking barplots for response variable vs each of explanatory variables

* Dropped features with unbalanced CI
  * Left one is bad, but Right one is good

* Dropped features: 'v24a_IT', 'v52', 'v54', 'v64', 'f96', 'v102', 'v129', 'v172', 'v184', 'v171', 'v215', 'v174_LR'

* Found this method causing more overfitting issues. (Mlog_loss was not improved.)

![image](https://github.com/jy-canucks1/KaggleComp_UW_STAT441/assets/84373345/03231e08-01bc-4c8b-944f-c750f4e8e8ed) ![image](https://github.com/jy-canucks1/KaggleComp_UW_STAT441/assets/84373345/9f8dd9ec-6ba2-4539-8582-ab9ac9d0cf76)

#### One Hot Encoding

* Removed related Numerical type of variables and performed one hot encoding to String (str4) type of variables (ex: v228b, v231b, v233b, 251b, ...)
* Used get_dummies from Pandas
* Example variable: v228b (respondents country of birth)
  
![image](https://github.com/jy-canucks1/KaggleComp_UW_STAT441/assets/84373345/45945086-365f-4969-bbf5-f9abbd57060d)

#### Final EDA version used
* Columns with NA values → Imputation with ‘-3’
* Dropped country specific columns (*_GB, *_DE)
* Merge Columns
* Time related conversion
* Dropped Variables with correlation value over 0.95 to lower overfitting
  
Total # of columns: 318

![image](https://github.com/jy-canucks1/KaggleComp_UW_STAT441/assets/84373345/e3c00728-4cb7-417c-a909-c4b4278f3cfe)


#### Model Building
* Model
  * Neural Network
  * LightGBM
  * RandomForest
  * SVM
  * XGBoost
* Hyperparameter tuning
  * GridSearchCV
  * Bayesian Optimization

#### Model Selection
* Model for testing
  * 3 Hidden layers with sigmoid activation function
  * LogSoftmax was applied in output layer
  * Trained with SGD
  * Loss function : CrossEntropyLoss (not mlog_loss)
  * Unstable learning curve with training set => bad model
Kaggle Public Score: 1.42958
Kaggle Private Score: 1.42854

![image](https://github.com/jy-canucks1/KaggleComp_UW_STAT441/assets/84373345/178e9f3f-382a-4db9-8f5c-df480ee765b6)


