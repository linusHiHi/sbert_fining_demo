import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizer, BertModel
import numpy as np

from util.dataset import ATECDataset, collate_fn
from util import EarlyStopping, save_checkpoint_torch
from util.util import convert_excel_to_classification_format

# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRE_TRAIN_PATH = "./bert-base-chinese"
PATH_TO_WHOLE_MODEL = "./miaModel/whole.pth"
PATH_TO_BERT_MODEL = "./miaModel/bert.pth"
BATCH_SIZE = 32
EPOCHS = 3
# 加载预训练模型
TOKENIZER = BertTokenizer.from_pretrained(PRE_TRAIN_PATH)
PRE_TRAIN_CONFIG = BertConfig.from_pretrained(PRE_TRAIN_PATH)
PRE_TRAIN = BertModel.from_pretrained(PRE_TRAIN_PATH)

# Sentence-BERT模型
class SentenceBert(nn.Module):
    def __init__(self):
        super(SentenceBert, self).__init__()
        self.pre_train = PRE_TRAIN
        self.linear = nn.Linear(PRE_TRAIN_CONFIG.hidden_size * 3, 2)
        nn.init.xavier_normal_(self.linear.weight.data)

    def get_cosine_score(self, s1, s2):
        s1_norm = s1 / torch.norm(s1, dim=1, keepdim=True)
        s2_norm = s2 / torch.norm(s2, dim=1, keepdim=True)
        cosine_score_ = (s1_norm * s2_norm).sum(dim=1)
        return cosine_score_

    def forward(self, x):
        s_emb = self.pre_train(**x)
        s_emb = s_emb['last_hidden_state'][:, 0, :]
        s1_emb, s2_emb = s_emb[::2], s_emb[1::2]

        concat = torch.concat([s1_emb, s2_emb, torch.abs(s1_emb - s2_emb)], dim=1)
        output_ = self.linear(concat)

        cosine_score_ = self.get_cosine_score(s1_emb, s2_emb)
        return output_, cosine_score_

# 计算皮尔逊相关系数
def compute_corrcoef(predictions_, labels_):
    predictions_, labels_ = np.array(predictions_), np.array(labels_)
    return np.corrcoef(predictions_, labels_)[0, 1]

# 验证函数
def eval_metrics(model_, loader):
    model_.eval()
    loss_sum, corrcoef_sum, count = 0, 0, 0
    with torch.no_grad():
        for s, labels in loader:
            s, labels = s.to(DEVICE), labels.to(DEVICE)[::2]
            output, cosine_score = model_(s)
            loss = criterion(output, labels)
            loss_sum += loss.item()
            corrcoef_sum += compute_corrcoef(cosine_score.detach().cpu().numpy(), labels.detach().cpu().numpy())
            count += 1
    return loss_sum / count, corrcoef_sum / count

# 数据加载
data_excel = pd.read_excel("./data/data.xlsx", sheet_name=None)
data = convert_excel_to_classification_format(data_excel)
"""
data = [
    ["打不开花呗", "为什么花呗打不开", 1],
    ["花呗收钱就是用支付宝帐号收嘛", "我用手机花呗收钱", 0],
    ["花呗买东西，商家不发货怎么退款", "花呗已经分期的商品 退款怎么办", 0]
]"""
train_loader = DataLoader(ATECDataset(data), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(ATECDataset(data), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)



# 模型、损失函数和优化器
model = SentenceBert().to(DEVICE)
criterion = nn.CrossEntropyLoss(reduction="mean")
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00003)

# 训练
early_stop = EarlyStopping(patience=1, delta=0.01, path_bert=PATH_TO_BERT_MODEL, path_whole=PATH_TO_WHOLE_MODEL)
for epoch in range(EPOCHS):
    model.train()
    for step, (s, labels) in enumerate(train_loader):
        s, labels = s.to(DEVICE), labels.to(DEVICE)[::2]
        optimizer.zero_grad()
        output, cosine_score = model(s)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            corrcoef = compute_corrcoef(cosine_score.detach().cpu().numpy(), labels.detach().detach().cpu().numpy())
            print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item()}, Corrcoef: {corrcoef}")

    loss_val, corrcoef_val = eval_metrics(model, val_loader)
    if early_stop(corrcoef_val, model):
        print("Early stopped")
        break

    print(f"Validation - Epoch {epoch+1}, Loss: {loss_val}, Corrcoef: {corrcoef_val}")

# 保存模型
save_checkpoint_torch(model, PATH_TO_WHOLE_MODEL, PATH_TO_BERT_MODEL)
# 文本匹配检索
from sklearn.metrics.pairwise import cosine_similarity
def search_top_n(input_text_, candidate_text, candidate_embeddings, top_n=3):
    text_token_ = TOKENIZER.batch_encode_plus([input_text_], truncation=True, padding=True,
                                              max_length=PRE_TRAIN_CONFIG.max_position_embeddings,
                                              return_tensors="pt")
    embeddings_ = PRE_TRAIN(**text_token_)[0][:, 0, :].detach().cpu().numpy()

    embeddings_ = embeddings_ / np.linalg.norm(embeddings_, axis=1)
    candidate_embeddings = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)

    # 使用sklearn的cosine_similarity
    scores = cosine_similarity(embeddings_, candidate_embeddings)
    top_index = np.argsort(scores, axis=1)[:, -top_n:]

    res = [{"text": candidate_text[i], "score": scores[0, i]} for i in top_index[0]]
    return res

# 测试
input_text = "没网的时候支付宝能够支付吗"
candidate_texts = [d[0] for d in data] + [d[1] for d in data]
candidate_emb = []

for text in candidate_texts:
    text_token = TOKENIZER.batch_encode_plus([text], truncation=True, padding=True,
                                             max_length=PRE_TRAIN_CONFIG.max_position_embeddings,
                                             return_tensors="pt")
    embeddings = PRE_TRAIN(**text_token)[0][:, 0, :].detach().cpu().numpy()
    candidate_emb.append(embeddings[0])

results = search_top_n(input_text, candidate_texts, candidate_emb, top_n=3)
print(f"\n\n\n\n{results}")
