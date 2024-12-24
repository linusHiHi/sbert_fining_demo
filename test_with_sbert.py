import numpy as np
from sentence_transformers import SentenceTransformer, LoggingHandler
import logging
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

PRE_TRAIN_PATH = "sbert_result"
ERROR_INPUT = "./data/found_class_3.csv"
train_batch_size = 16
num_epochs = 4
logging.basicConfig(format='%(asctime)s - %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S',
  level=logging.INFO,
  handlers=[LoggingHandler()])

# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings

model = SentenceTransformer(PRE_TRAIN_PATH)
"""
def sampling(datas):

  samples = []
  for item in datas:
    samples.append(InputExample(texts=[item[0], item[1]], label=float(item[2])))
  return samples
"""
data_excel = pd.read_excel("./data/source_data.xlsx", sheet_name=None)
data = []
for sheet_name, df in data_excel.items():
    sentences = df['sentence'].tolist()  # 获取当前类的所有句子
    data += sentences


# 测试新的输入文本
input_texts = ["我想去列车的车头看看"]



"""
input_texts = pd.read_csv("./data/found_class_3.csv")
input_texts = input_texts["sentence"]
"""



error_input = []

embeddings = model.encode(input_texts)

# 假设候选文本和它们的嵌入已经准备好
candidate_texts = data
candidate_embeddings = model.encode(candidate_texts)


# 计算余弦相似度
similarities = cosine_similarity(embeddings, candidate_embeddings)
similarities=np.array(similarities)
similarities=np.vstack(similarities).reshape(len(input_texts), len(candidate_texts))
# 输出前{top_n}个最相似的文本
top_n = 1
for i in range(len(input_texts)):
    top_indices = similarities.argsort()[i][-top_n:]
    print(f"original: {input_texts[i]}")
    for idx in top_indices:
        if similarities[i][idx] > 0.001:
            print(f"Text: {candidate_texts[idx]}, Similarity Score: {similarities[i][idx]}")
            error_input.append(input_texts[i])


try:
    error_df = pd.read_csv(ERROR_INPUT)
    error_df = pd.concat([error_df,pd.DataFrame({"sentence":error_input, "class":3})])
    error_df.to_csv(ERROR_INPUT, index=False)
except FileNotFoundError:
    error_df = pd.DataFrame({"error": error_input, "class":3})
    error_df.to_csv(ERROR_INPUT,index=False)
