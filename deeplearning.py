# %% Import Libraries
import os
import csv
import fasttext

import pandas as pd
import numpy as np

from gensim.utils import simple_preprocess
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from time import perf_counter

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

fasttext_parameters = {
    'lr': 1.0,
    'epoch': 25,
    'wordNgrams': 2, # bigrams
    'bucket':2_000_000,
    'dim': 100,
    'loss': 'hs' # hierarchical softmax
}

# Train
dl_train_time_start = perf_counter()
dl_model = fasttext.train_supervised(os.path.join(__BASE_PATH, OUTPUT_FOLDER, FASTTEXT_TRAIN_FILENAME), **fasttext_parameters)
dl_train_time_stop = perf_counter()
print(f"Elapsed time: {dl_train_time_stop - dl_train_time_start}")
print(f"Number of words: {len(dl_model.words)}")
print(f"Number of labels: {len(dl_model.labels)}")
# %% Test
fasttext_label_list = fasttext_preprocessed_test['label'].values.tolist()

dl_prediction = dl_model.predict(
    fasttext_preprocessed_test['text_a'].values.tolist(),
    k=1
)

dl_prediction_flat = [item for sublist in dl_prediction[0] for item in sublist]
print(f"[Deeplearning] Accuracy: {accuracy_score(fasttext_label_list, dl_prediction_flat)}")
print(f"[Deeplearning] Precision: {precision_score(fasttext_label_list, dl_prediction_flat, pos_label='__label__yes')}")
print(f"[Deeplearning] Recall: {recall_score(fasttext_label_list, dl_prediction_flat, pos_label='__label__yes')}")
print(f"[Deeplearning] F1-score: {f1_score(fasttext_label_list, dl_prediction_flat, pos_label='__label__yes')}")
# %% Predict
dl_model.predict(TEST_PREDICT_TEXT, k=1)
