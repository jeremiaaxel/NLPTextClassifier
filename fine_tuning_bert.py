# %% Import Libraries
import os
import torch
import numpy as np
import evaluate
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# %% Configurations
TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"
LABEL_FILENAME = "labels.txt"
INPUT_FOLDER_PATH = "drive/Shareddrives/NLP/Text Classifier/data_worthcheck"  # For Google colab

FASTTEXT_TRAIN_FILENAME = "train.txt"
FASTTEXT_TEST_FILENAME = "test.txt"
OUTPUT_FOLDER = "out"

TEST_PREDICT_TEXT = "Josep ganteng anjay"

__BASE_PATH = os.getcwd()
__FULL_OUTPUT_FOLDER = os.path.join(__BASE_PATH, OUTPUT_FOLDER)
if not os.path.exists(__FULL_OUTPUT_FOLDER):
	os.makedirs(__FULL_OUTPUT_FOLDER)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

BERT_PRETRAINED_MODEL_PATH = "indobenchmark/indobert-base-p2"

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


def tokenize(queries):
	tokenized_queries = [
		{**tokenizer(query, truncation=True, padding='max_length', max_length=280, return_tensors='pt'),
		 'labels': 0 if train_labels[i] == 'no' else 1} for (i, query) in enumerate(queries)]
	# Squeeze the tensors into a 1D tensor
	return [{key: toked[key].squeeze(0) if torch.is_tensor(toked[key]) else toked[key] for key in toked} for toked in
			tokenized_queries]


tokenized_train_data = tokenize(train_queries)
tokenized_test_data = tokenize(test_queries)

tokenized_train_data[0]

# %% Create model
model = BertForSequenceClassification.from_pretrained(BERT_PRETRAINED_MODEL_PATH, num_labels=2)
model.to(device)

# %% fine-tune model
torch.cuda.empty_cache()


def compute_metrics(eval_pred):
	logits, labels = eval_pred
	predictions = np.argmax(logits, axis=-1)
	return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
	evaluation_strategy='epoch',
	logging_steps=200,
	per_device_train_batch_size=16,
	output_dir='./bert-output',
	weight_decay=0.01,
)
trainer = Trainer(
	model=model,
	args=training_args,
	train_dataset=tokenized_train_data,
	eval_dataset=tokenized_test_data,
	compute_metrics=compute_metrics,
	tokenizer=tokenizer,
)
trainer.train()

# %%
model(**tokenized_test_data[0])
