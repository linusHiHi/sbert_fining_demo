
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer, LoggingHandler, losses, models

import logging

import pandas as pd
import math

from util import split_data, convert_util, sampling
"""
******************************************************
"""
model_name = 'bert-base-chinese'
model_save_path = 'sbert_result'
path_to_whole_dataset = "data/full_data_with_all_new_sentence.csv"
path_to_raw_excel_dataset = "data/source_data.xlsx"
train_batch_size = 16
num_epochs = 4
warm_up_rate = 0.1
classes = 4
evaluation_steps = 1000
evaluate_name = 'sts-dev'
test_name = 'sts-test'


logging.basicConfig(format='%(asctime)s - %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S',
  level=logging.INFO,
  handlers=[LoggingHandler()])

"""
*******************Model defining**************************
"""

# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(
  word_embedding_model.get_word_embedding_dimension(),
  pooling_mode_mean_tokens=True,
  pooling_mode_cls_token=False,
  pooling_mode_max_tokens=False
)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


"""
******************data dealing**************************
"""

"""
data_excel = pd.read_excel(path_to_raw_excel_dataset, sheet_name=None)
data = convert_excel_to_classification_format(data_excel,"excel")
"""
data_csv = pd.read_csv(path_to_whole_dataset)
data = convert_util(data_csv,"csv",classes)

"""
train_samples, test_samples, dev_samples = split_data(data)

train_samples = sampling(train_samples)
dev_samples = sampling(dev_samples)
test_samples = sampling(test_samples)
"""

train_samples, test_samples, dev_samples = split_data(data).map(sampling)

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name=evaluate_name)
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * warm_up_rate) #10% of train data for warm-up

"""
*********************Train the model*******************************
"""
model.fit(train_objectives=[(train_dataloader, train_loss)],
  evaluator=evaluator,
  epochs=num_epochs,
  evaluation_steps=evaluation_steps,
  warmup_steps=warmup_steps,
  output_path=model_save_path)

"""
*****************  test  ******************
"""

model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name=test_name)
test_evaluator(model)
