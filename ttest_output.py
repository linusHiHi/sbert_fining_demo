import numpy as np
from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers.readers import InputExample
import logging
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

PRE_TRAIN_PATH = "./test_output"
train_batch_size = 16
num_epochs = 4
logging.basicConfig(format='%(asctime)s - %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S',
  level=logging.INFO,
  handlers=[LoggingHandler()])

# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings

model = SentenceTransformer(PRE_TRAIN_PATH)

def sampling(datas):

  samples = []
  for item in datas:
    samples.append(InputExample(texts=[item[0], item[1]], label=float(item[2])))
  return samples

data_excel = pd.read_excel("./data/data.xlsx", sheet_name=None)
data = []
for sheet_name, df in data_excel.items():
    sentences = df['sentence'].tolist()  # 获取当前类的所有句子
    data += sentences


# 测试新的输入文本
input_texts = "千问"
embeddings = model.encode(input_texts)

# 假设候选文本和它们的嵌入已经准备好
#candidate_texts = data
candidate_texts = ["汉堡","哈布斯堡","谢林", "引力波","火车票","冰淇淋","果冻","面包"]
candidate_embeddings = model.encode(candidate_texts)

# 计算余弦相似度
similarities = [cosine_similarity(embeddings.reshape(1,-1), candidate_embedding.reshape(1,-1)) for candidate_embedding in candidate_embeddings]
similarities=np.array(similarities)
similarities=np.vstack(similarities).reshape(-1)
# 输出前{top_n}个最相似的文本
top_n = 5
top_indices = similarities.argsort()[0:top_n]
last = similarities.argsort()[-top_n:-1]
print(f"original: {input_texts}")
for idx in top_indices:
    print(f"Text: {candidate_texts[idx]}, Similarity Score: {similarities[idx]}")
for idx in last:
    print(f"fail Text: {candidate_texts[idx]}, Similarity Score: {similarities[idx]}")
