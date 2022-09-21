# %% Import Libraries
import os
import csv

import pandas as pd
import numpy as np

import fasttext
from gensim.utils import simple_preprocess

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

# %% Deep learning: fasttext
# https://fasttext.cc/docs/en/python-module.html

# Train
dl_model = fasttext.train_supervised(os.path.join(__BASE_PATH, OUTPUT_FOLDER, FASTTEXT_TRAIN_FILENAME))
# %% Test
dl_n_test, dl_precision, dl_recall = dl_model.test(os.path.join(__BASE_PATH, OUTPUT_FOLDER, FASTTEXT_TEST_FILENAME))
print(f"[Deeplearning] Number of tests: {dl_n_test}")
print(f"[Deeplearning] Precision: {dl_precision}")
print(f"[Deeplearning] Recall: {dl_recall}")
# %% Predict
dl_model.predict(TEST_PREDICT_TEXT)