from torch.utils.data import Dataset
from transformers import BertTokenizer, BertConfig, BertModel


import itertools
from sklearn.model_selection import train_test_split
import torch
from sentence_transformers import InputExample
import random

PRE_TRAIN_PATH = "./bert-base-chinese"

TOKENIZER = BertTokenizer.from_pretrained(PRE_TRAIN_PATH)
PRE_TRAIN_CONFIG = BertConfig.from_pretrained(PRE_TRAIN_PATH)
PRE_TRAIN = BertModel.from_pretrained(PRE_TRAIN_PATH)
# 数据集类
class ATECDataset(Dataset):
    def __init__(self, data_):
        self.data = data_

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 数据处理函数
def collate_fn(data_):
    s_, labels_ = [], []
    for d in data_:  # 三元组(s1, s2, label)
        s_.append(d[0])
        s_.append(d[1])
        labels_.append(d[2])
        labels_.append(d[2])
    s_token = TOKENIZER.batch_encode_plus(s_, truncation=True, max_length=PRE_TRAIN_CONFIG.max_position_embeddings,
                                          return_tensors="pt", padding=True)
    labels_ = torch.LongTensor(labels_)
    return s_token, labels_



"""
******************* convert utils ***********************
"""
def convert_util(data_, form, classes=None):
    if form == 'excel':
        return convert_excel_to_classification_format(data_)
    elif form == 'csv':
        if classes is None:
            raise TypeError('classes cannot be None')
        return convert_csv_to_classification_format(data_, classes)
    else:
        raise NotImplementedError
def dic_to_classification_format(sentences_by_class):
    # 处理每个类的句子
    data = []
    for class_name, sentences in sentences_by_class.items():
        # 将相同类中的句子两两组合，并添加标签 1
        for sentence1, sentence2 in itertools.combinations(sentences, 2):
            data.append([sentence1, sentence2, 1])

        """
        # 处理不同类之间的句子对
        for other_class_name, other_sentences in sentences_by_class.items():
            if int(other_class_name) > int(class_name):
                for sentence1 in sentences:
                      for sentence2 in other_sentences:
                        if [sentence1, sentence2, 0] not in data and [sentence2, sentence1, 0] not in data:
                            data.append([sentence1, sentence2, 0])
        """



        # 假设每个类别的句子保存在 sentences_by_class 字典中
        # 采样参数
        sample_size =270  # 从每个类别中随机选择的句子数目

        # 处理不同类之间的句子对
        for other_class_name, other_sentences in sentences_by_class.items():
            if int(other_class_name) > int(class_name):
                # 随机采样一定数量的句子（如果该类别的句子数目大于 sample_size）
                sampled_sentences = random.sample(list(sentences), min(sample_size, len(sentences)))
                sampled_other_sentences = random.sample(list(other_sentences), min(sample_size, len(other_sentences)))

                for sentence1 in sampled_sentences:
                    for sentence2 in sampled_other_sentences:
                        data.append([sentence1, sentence2, 0])

    return data


def convert_excel_to_classification_format(sheets):
    # 读取 Excel 文件中的所有工作表

    # 用于存储所有的句子对
    sentences_by_class = {}

    # 读取每个 sheet 中的句子数据
    for sheet_name, df in sheets.items():
        sentences = df['sentence'].tolist()  # 获取当前类的所有句子
        sentences_by_class[sheet_name] = sentences

    return dic_to_classification_format(sentences_by_class)




def convert_csv_to_classification_format(df, classes):
    sentences_by_class = {}
    for i in range(classes):
        sentences_by_class[f"{i}"] = df["sentence"][df["class"]==i]
    return dic_to_classification_format(sentences_by_class)

"""
***************** split data *********************
"""


def split_data(data, train_size=0.7, val_size=0.15, test_size=0.15, random_state=None):
    # 首先，划分训练集和剩余集（验证集 + 测试集）
    train_data, temp_data = train_test_split(data, train_size=train_size, random_state=random_state)

    # 然后，将剩余集划分为验证集和测试集
    val_data, test_data = train_test_split(temp_data, train_size=val_size / (val_size + test_size), random_state=random_state)

    return train_data, val_data, test_data


"""
***************** sampling data *********************
"""


def sampling(datas):
  samples = []
  for item in datas:
    samples.append(InputExample(texts=[item[0], item[1]], label=float(item[2])))
  return samples