# %% Import Libraries
import pandas as pd
import numpy as np

# Shallow Machine Learning
from sklearn.svm import LinearSVR # SVM
import xgboost as xgb # XGBOOST

# Deep learning: Non-contextual word embedding
from gensim.models import Word2Vec # Word2Vec

# BERT (fine-tuning?)
# https://pypi.org/project/fast-bert/
# !pip install fast-bert==1.9.1 # uncomment to manually install fast_bert
from fast_bert.data_cls import BertDataBunch
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy

# %%




