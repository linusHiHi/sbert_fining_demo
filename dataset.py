import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertConfig, BertModel

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
