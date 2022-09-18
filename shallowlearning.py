# %% Import Libraries
import os
import csv

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, accuracy_score
import xgboost as xgb

# %% Configurations

TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"
LABEL_FILENAME = "labels.txt"
INPUT_FOLDER_PATH = "data_worthcheck"

FASTTEXT_TRAIN_FILENAME = "train.txt"
FASTTEXT_TEST_FILENAME = "test.txt"
OUTPUT_FOLDER = "out"

TEST_PREDICT_TEXT = "Josep ganteng anjay"

__BASE_PATH = os.getcwd()
__FULL_OUTPUT_FOLDER = os.path.join(__BASE_PATH, OUTPUT_FOLDER)
if not os.path.exists(__FULL_OUTPUT_FOLDER):
    os.makedirs(__FULL_OUTPUT_FOLDER)

# %% Prepare Datasets
train_df = pd.read_csv(os.path.join(__BASE_PATH, INPUT_FOLDER_PATH, TRAIN_FILENAME))
test_df = pd.read_csv(os.path.join(__BASE_PATH, INPUT_FOLDER_PATH, TEST_FILENAME))

train_df.info()

# %% Preprocessing
svm_preprocessed_train = train_df[['text_a', 'label']]
svm_preprocessed_test = test_df[['text_a', 'label']]

# %% Shallow learning: SVM
# https://ai.plainenglish.io/sentiment-classification-using-xgboost-7abdaf4771f9
# https://towardsdatascience.com/basics-of-countvectorizer-e26677900f9c

# Remove stopwords
stopwordlist = stopwords.words('indonesian')
svm_pre_train_x = []
svm_pre_test_x = []
for sentence in svm_preprocessed_train['text_a']:
    svm_pre_train_x.append(' '.join([word for word in sentence.split() if word not in stopwordlist]))
for sentence in svm_preprocessed_test['text_a']:
    svm_pre_test_x.append(' '.join([word for word in sentence.split() if word not in stopwordlist]))

# Vectorize sentences
cv = CountVectorizer()
cv.fit(svm_pre_train_x)
svm_train_x = cv.transform(svm_pre_train_x)
svm_test_x = cv.transform(svm_pre_test_x)

# Convert yes to 1 and no to 0
xgb_train_labels = []
for label in svm_preprocessed_train['label']:
    if label == "yes":
        xgb_train_labels.append(1)
    elif label == "no":
        xgb_train_labels.append(0)
    else:
        xgb_train_labels.append(None)

xgb_test_labels = []
for label in svm_preprocessed_test['label']:
    if label == "yes":
        xgb_test_labels.append(1)
    elif label == "no":
        xgb_test_labels.append(0)
    else:
        xgb_test_labels.append(None)

# Train
xgb_train = xgb.DMatrix(svm_train_x, xgb_train_labels)
xgb_test = xgb.DMatrix(svm_test_x, xgb_test_labels)
svm_param = {'eta': 0.75,
         'max_depth': 50,
         'objective': 'binary:logitraw',
         'eval_metric': 'logloss'}
xgb_model = xgb.train(svm_param, xgb_train, num_boost_round = 30)

# %% Test
svm_y_pred = xgb_model.predict(xgb_test)
svm_y_pred = np.where(np.array(svm_y_pred) > 0.5, 1, 0)
print(svm_param)
print("Test data prediction accuracy score: ", accuracy_score(xgb_test_labels, svm_y_pred))
print("Test data prediction f1 score      : ", f1_score(xgb_test_labels, svm_y_pred))

# %%
