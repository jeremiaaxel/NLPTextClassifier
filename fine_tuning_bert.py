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
INPUT_FOLDER_PATH = os.path.join("data_worthcheck")
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


# %% Helper methods
def map_label_to_id(label):
	return 1 if label == 'yes' else 0


def label_to_id(labels):
	return [map_label_to_id(label) for label in labels]


def map_id_to_label(id):
	return 'yes' if id == 1 else 0


def id_to_label(ids):
	return [map_id_to_label(id) for id in ids]


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


def tokenize(queries, squeeze=True, labels=None):
	tokenizer_args = {
		'truncation': True,
		'padding': 'max_length',
		'max_length': 280,
		'return_tensors': 'pt'
	}
	if labels is not None:
		toked_queries = [
			{**tokenizer(query, **tokenizer_args), 'labels': map_label_to_id(labels[i])} for (i, query) in
			enumerate(queries)]
	else:
		toked_queries = [tokenizer(query, **tokenizer_args) for query in queries]
	if squeeze:
		# Squeeze the tensors into a 1D tensor
		toked_queries = [{key: toked[key].squeeze(0) if torch.is_tensor(toked[key]) else toked[key] for key in toked}
						 for toked in toked_queries]

	return toked_queries


tokenized_train_data = tokenize(train_queries, labels=train_labels)
tokenized_validation_data = tokenize(validation_queries, labels=validation_labels)
tokenized_test_data = tokenize(test_queries, squeeze=False)
# Set test data to use CUDA if available
if torch.cuda.is_available():
	for data in tokenized_test_data:
		for k in data:
			data[k].to(device)

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

# %% Test the model
results = []
counter = 1
with torch.no_grad():
	for test_data in tokenized_test_data:
		output = model(**test_data)
		final_output = torch.sigmoid(output.logits).cpu().detach().numpy().tolist()
		result = map_id_to_label(np.argmax(final_output, axis=1)[0])
		print(f'{counter}-th data:', result, final_output)
		results.append(result)

results

# %% Evaluate testing results
accuracy_result = metric.compute(references=test_labels, predictions=results)
print('Accuracy:', accuracy_result['accuracy'])
