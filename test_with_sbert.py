import numpy as np
from sentence_transformers import SentenceTransformer, LoggingHandler
import logging
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

CLASS_OTHER = 3
topN = 5

"""
***************** config **********************
"""
PRE_TRAIN_PATH = "./sbert_result"
ERROR_INPUT = "./data/found_class_0_3.csv"
CANDIDATE_INPUT = "./data/sampled_data_with_all_new_sentence.csv"
train_batch_size = 16
num_epochs = 4
logging.basicConfig(format='%(asctime)s - %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S',
  level=logging.INFO,
  handlers=[LoggingHandler()])

# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
accept_rate = 0.75
model = SentenceTransformer(PRE_TRAIN_PATH)

candidate_texts = pd.read_csv(CANDIDATE_INPUT)

"""
*************** input data ************************
"""
# 测试新的输入文本
input_texts = ["黑格尔坐马车去柏林吗", "那我问你，笛卡尔有没有坐过火车？", "你听过春卷饭的新专吗"]
input_class_tag = 3
# input_texts = ["点餐", "今天天气如何", "我想去列车的车头看看"]
# input_class_tag = [1,2,3]
"""
input_texts = pd.read_csv("./data/found_class_0_3.csv")
input_texts = input_texts["sentence"]
"""

"""
***************** 测试 **********************
"""
class ClassTag():
    def __init__(self, tag):
        self.tag = tag
        if type(tag) == int:
            self.cast = True
        else:
            self.cast = False
    def __call__(self, index_):
        if self.cast:
            return self.tag
        else:
            return self.tag[index_]

def test(input_texts_, candidate_tests_, accept_rate_, input_class_tag_=3, top_n = 1):
    input_class_tag_ = ClassTag(input_class_tag_)

    error_input = []
    error_input_class = []
    embeddings = model.encode(input_texts_)

    # 假设候选文本和它们的嵌入已经准备好
    candidate_embeddings = model.encode(candidate_tests_["sentence"])


    # 计算余弦相似度
    similarities = cosine_similarity(embeddings, candidate_embeddings)
    """
    similarities=np.array(similarities)
    similarities=np.vstack(similarities).reshape(len(input_texts), len(candidate_tests_))
    """
    # 输出前{top_n}个最相似的文本

    for i in range(len(input_texts_)):
        top_indices = similarities.argsort()[i][-top_n:]
        print(f"original: {input_texts_[i]}, class: {input_class_tag_(i)}")
        flag_write = False
        for idx in top_indices:
            """
            大于置信度说明positive，然后判断true false
            小于置信度说明没有匹配上，则可能是3匹配正常，也可能是tn
            """

            if similarities[i][idx] > accept_rate_:
                if candidate_tests_["class"][idx] == input_class_tag_(i):
                    print(f"Text: {candidate_tests_["sentence"][idx]}, "
                          f"Similarity Score: {similarities[i][idx]}, "
                          f"class: {input_class_tag_(i)}")
                else:
                    print(f"ERROR MATCHED "
                          f"Text: {candidate_tests_["sentence"][idx]}, "
                          f"Similarity Score: {similarities[i][idx]}, "
                          f"matched: {candidate_tests_["class"][idx]}")
                    error_input.append(input_texts_[i])
                    error_input_class.append(input_class_tag_(int(i)))
            else:
                if input_texts_[i] != CLASS_OTHER:
                    print(f"ERROR UNMATCHED "
                          f"Text: {candidate_tests_["sentence"][idx]}, "
                          f"Similarity Score: {similarities[i][idx]}, "
                          f"UNMATCHED: {candidate_tests_["class"][idx]}")
                    error_input.append(input_texts_[i])
                    error_input_class.append(input_class_tag_(int(i)))
                else:
                    print(f"CAUGHT OTHER "
                          f"Text: {candidate_tests_["sentence"][idx]}, "
                          f"Similarity Score: {similarities[i][idx]}, "
                          f"SOURCE: {candidate_tests_["class"][idx]}")


    return pd.DataFrame({"sentence":error_input, "class":error_input_class})


test_df = test(input_texts, candidate_texts, accept_rate, input_class_tag_=input_class_tag,top_n=topN)

try:
    error_df = pd.read_csv(ERROR_INPUT) # 这里可能还没有创建文件
    error_df = pd.concat([error_df,test_df])
    error_df.drop_duplicates(subset=["sentence"], inplace=True,keep="first")
    error_df["class"]= error_df["class"].astype("int32")
    error_df.to_csv(ERROR_INPUT, index=False)
except FileNotFoundError:
    test_df.to_csv(ERROR_INPUT,index=False)
