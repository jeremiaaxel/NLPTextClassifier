# %% Import Libraries
import os
import torch
import numpy as np
import evaluate
import pandas as pd
from transformers import BertTokenizer, BertForQuestionAnswering, Trainer, TrainingArguments

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

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

# BERT_PRETRAINED_MODEL_PATH = "indobenchmark/indobert-base-p2"
# BERT_PRETRAINED_MODEL_PATH = "bert-base-uncased"
BERT_PRETRAINED_MODEL_PATH = "bert-base-uncased"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
metric = evaluate.load('accuracy')

# %% Prepare Datasets
train_df = pd.read_csv(os.path.join(__BASE_PATH, INPUT_FOLDER_PATH, TRAIN_FILENAME))
test_df = pd.read_csv(os.path.join(__BASE_PATH, INPUT_FOLDER_PATH, TEST_FILENAME))

train_df.info()

# %% Get "queries" and labels
train_queries = list(train_df['text_a'])
train_labels = list(train_df['label'])

test_queries = list(test_df['text_a'])
test_labels = list(test_df['label'])

(train_queries[0], train_labels[0])

# %% Encode training data (create input IDs, token type IDs, and attention masks) in the form of tenors
tokenizer = BertTokenizer.from_pretrained(BERT_PRETRAINED_MODEL_PATH, do_lower_case=True)

train_encodings = [
	tokenizer(query, truncation=True, padding='max_length', max_length=280, return_tensors='pt') for
	(i, query) in enumerate(train_queries)]
# Squeeze the tensors into a 1D array
train_encodings = [{keys: encoding[keys].squeeze(0) for keys in encoding} for encoding in train_encodings]

test_encodings = [
	tokenizer(query, truncation=True, padding='max_length', max_length=280, return_tensors='pt') for
	(i, query) in enumerate(test_queries)]
test_encodings = [{keys: encoding[keys].squeeze(0) for keys in encoding} for encoding in test_encodings]

train_encodings[0]

# %% Create model
model = BertForQuestionAnswering.from_pretrained(BERT_PRETRAINED_MODEL_PATH, num_labels=2)
model.to(device)

# %% fine-tune model
def compute_metrics(eval_pred):
	logits, labels = eval_pred
	predictions = np.argmax(logits, axis=-1)
	return metric.compute(predictions=predictions, references=labels)


torch.cuda.empty_cache()
training_args = TrainingArguments(
	# num_train_epochs=10,
	# weight_decay=0.01,
	# load_best_model_at_end=True,
	# logging_steps=200,
	# save_steps=400,
	# evaluation_strategy='steps',
	evaluation_strategy='epoch',
	per_device_train_batch_size=4,
	output_dir='./bert-output',
)
trainer = Trainer(
	model=model,
	args=training_args,
	train_dataset=train_encodings,
	eval_dataset=test_encodings,
	compute_metrics=compute_metrics,
)
trainer.train()

# %%
model(**test_encodings[0])
