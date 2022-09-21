# %% Import Libraries
import os
import csv
import nltk
import fasttext
import pandas as pd
import numpy as np
import xgboost as xgb

from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, accuracy_score
from nltk.corpus import stopwords
# %% Download library-requirements
nltk.download('stopwords')
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

# %% ******** SHALLOW LEARNING - SVM ******* %% #
# https://ai.plainenglish.io/sentiment-classification-using-xgboost-7abdaf4771f9
# https://towardsdatascience.com/basics-of-countvectorizer-e26677900f9c
# %% Preprocessing
svm_preprocessed_train = train_df[['text_a', 'label']]
svm_preprocessed_test = test_df[['text_a', 'label']]

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

# %% Train
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
print(f"Test data prediction accuracy score: {accuracy_score(xgb_test_labels, svm_y_pred)}")
print(f"Test data prediction f1 score      : {f1_score(xgb_test_labels, svm_y_pred)}")

# %% ******** DEEP LEARNING - FAST TEXT ******* %% #
# https://fasttext.cc/docs/en/python-module.html
# %% Preprocess csv to txt for fasttext input
fasttext_train = train_df[['text_a', 'label']]
fasttext_test = test_df[['text_a', 'label']]

fasttext_preprocessed_train = fasttext_train.copy()
fasttext_preprocessed_test = fasttext_test.copy()

fasttext_preprocessed_train.iloc[:, 0] = fasttext_train.iloc[:, 0].apply(lambda x: ' '.join(simple_preprocess(x)))
fasttext_preprocessed_test.iloc[:, 0] = fasttext_test.iloc[:, 0].apply(lambda x: ' '.join(simple_preprocess(x)))

fasttext_preprocessed_train.iloc[:, 1] = fasttext_train.iloc[:, 1].apply(lambda x: ''.join('__label__' + x))
fasttext_preprocessed_test.iloc[:, 1] = fasttext_test.iloc[:, 1].apply(lambda x: ''.join('__label__' + x))

fasttext_preprocessed_train[['label', 'text_a']].to_csv(
    os.path.join(__BASE_PATH, OUTPUT_FOLDER, FASTTEXT_TRAIN_FILENAME),
    index = False,
    sep = ' ',
    header = None,
    quoting = csv.QUOTE_NONE, 
    quotechar = "", 
    escapechar = " "
)

fasttext_preprocessed_test[['label', 'text_a']].to_csv(
    os.path.join(__BASE_PATH, OUTPUT_FOLDER, FASTTEXT_TEST_FILENAME),
    index = False,
    sep = ' ',
    header = None,
    quoting = csv.QUOTE_NONE, 
    quotechar = "", 
    escapechar = " "
)

# %% Train
dl_model = fasttext.train_supervised(os.path.join(__BASE_PATH, OUTPUT_FOLDER, FASTTEXT_TRAIN_FILENAME))

# %% Test
dl_n_test, dl_precision, dl_recall = dl_model.test(os.path.join(__BASE_PATH, OUTPUT_FOLDER, FASTTEXT_TEST_FILENAME))
print(f"[Deeplearning] Number of tests  : {dl_n_test}")
print(f"[Deeplearning] Precision        : {dl_precision}")
print(f"[Deeplearning] Recall: {dl_recall}")

# %% Predict
dl_model.predict(TEST_PREDICT_TEXT)
