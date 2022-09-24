# %% Import Libraries
import os
import torch
import numpy as np
import evaluate
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# %% Configurations
# Setup some constants
COLAB_PREFIX = os.path.join('drive', 'Shareddrives', 'NLP', 'Text Classifier')
TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"
LABEL_FILENAME = "labels.txt"
INPUT_FOLDER_PATH = os.path.join(COLAB_PREFIX, "data_worthcheck")
FASTTEXT_TRAIN_FILENAME = "train.txt"
FASTTEXT_TEST_FILENAME = "test.txt"
OUTPUT_FOLDER = os.path.join(COLAB_PREFIX, "out")
__BASE_PATH = os.getcwd()
__FULL_OUTPUT_FOLDER = os.path.join(__BASE_PATH, OUTPUT_FOLDER)

# Create output directory if not exist
if not os.path.exists(__FULL_OUTPUT_FOLDER):
	os.makedirs(__FULL_OUTPUT_FOLDER)

BERT_PRETRAINED_MODEL_PATH = "indobenchmark/indobert-base-p2"

# Set PyTorch device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# Set fine-tuning metric
metric = evaluate.load('accuracy')

# %% Prepare Datasets
train_df = pd.read_csv(os.path.join(__BASE_PATH, INPUT_FOLDER_PATH, TRAIN_FILENAME))
test_df = pd.read_csv(os.path.join(__BASE_PATH, INPUT_FOLDER_PATH, TEST_FILENAME))

train_df.info()

# %% Get "queries" and labels
train_queries = list(train_df['text_a'])
train_labels = list(train_df['label'])

train_queries, validation_queries, train_labels, validation_labels = train_test_split(train_queries, train_labels,
																					  test_size=0.2, random_state=2022)

test_queries = list(test_df['text_a'])
test_labels = list(test_df['label'])

(train_queries[0], train_labels[0])

# %% Tokenized training data (create input IDs, token type IDs, and attention masks) in the form of tensors
tokenizer = BertTokenizer.from_pretrained(BERT_PRETRAINED_MODEL_PATH, do_lower_case=True)


def tokenize(queries, labels):
	toked_queries = [
		{**tokenizer(query, truncation=True, padding='max_length', max_length=280, return_tensors='pt'),
		 'labels': 0 if labels[i] == 'no' else 1} for (i, query) in enumerate(queries)]
	# Squeeze the tensors into a 1D tensor
	return [{key: toked[key].squeeze(0) if torch.is_tensor(toked[key]) else toked[key] for key in toked} for toked in
			toked_queries]


tokenized_train_data = tokenize(train_queries, train_labels)
tokenized_validation_data = tokenize(validation_queries, validation_labels)
tokenized_test_data = tokenize(test_queries, test_labels)

tokenized_train_data[0]

# %% Create model
model = BertForSequenceClassification.from_pretrained(BERT_PRETRAINED_MODEL_PATH, num_labels=2)
model.to(device)

# %% fine-tune model
# Empty CUDA cache
torch.cuda.empty_cache()


def compute_metrics(eval_pred):
	"""
	the-compute metric method for trainer
	Credits to: https://huggingface.co/docs/transformers/training
	"""
	logits, labels = eval_pred
	predictions = np.argmax(logits, axis=-1)
	return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
	evaluation_strategy='epoch',
	per_device_train_batch_size=16,
	output_dir=__FULL_OUTPUT_FOLDER,
	weight_decay=0.01,
)
trainer = Trainer(
	model=model,
	args=training_args,
	train_dataset=tokenized_train_data,
	eval_dataset=tokenized_validation_data,
	compute_metrics=compute_metrics,
	tokenizer=tokenizer,
)
trainer.train()

# %%
model(**tokenized_test_data[0])
