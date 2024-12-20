from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
import logging
import pandas as pd
from transformers import BertModel

from util import convert_excel_to_classification_format, split_data
PRE_TRAIN_PATH = "./test_output"
train_batch_size = 16
num_epochs = 4
logging.basicConfig(format='%(asctime)s - %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S',
  level=logging.INFO,
  handlers=[LoggingHandler()])

# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings

PRE_TRAIN = SentenceTransformer(PRE_TRAIN_PATH)

def sampling(datas):

  samples = []
  for item in datas:
    samples.append(InputExample(texts=[item[0], item[1]], label=float(item[2])))
  return samples

data_excel = pd.read_excel("./data/data.xlsx", sheet_name=None)
data = convert_excel_to_classification_format(data_excel)

train_samples, test_samples, dev_samples = split_data(data)
"""train_samples = sampling(train_samples)
dev_samples = sampling(dev_samples)"""
test_samples = sampling(test_samples)
# test_loader = DataLoader(test_samples, shuffle=True, batch_size=train_batch_size)
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-dev')
res = evaluator(PRE_TRAIN)
print(res)