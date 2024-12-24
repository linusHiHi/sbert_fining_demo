import pandas as pd
from nltk.sentiment.util import split_train_test
from scipy.stats import pearsonr

import torch.nn as nn
from torch import cosine_similarity, device, cuda, abs, no_grad, optim, cat
from torch.utils.data import DataLoader

from transformers import BertConfig, BertTokenizer, BertModel

from util import EarlyStopping, save_checkpoint_torch, convert_util
from util.dataset import ATECDataset, collate_fn

# 配置
DEVICE = device("cuda" if cuda.is_available() else "cpu")
PRE_TRAIN_PATH = "./bert-base-chinese"
PATH_TO_WHOLE_MODEL = "./pytorch_result/whole.pth"
PATH_TO_BERT_MODEL = "./pytorch_result/bert.pth"
path_to_csv_dataset = "data/source_data.csv"
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


    def forward(self, x):
        s_emb = self.pre_train(**x)
        s_emb = s_emb['last_hidden_state'][:, 0, :]
        s1_emb, s2_emb = s_emb[::2], s_emb[1::2]

        concat = cat([s1_emb, s2_emb, abs(s1_emb - s2_emb)], dim=1)
        output_ = self.linear(concat)

        # cosine_score_ = self.get_cosine_score(s1_emb, s2_emb)
        cosine_score_ = cosine_similarity(s1_emb, s2_emb)
        return output_, cosine_score_

    """
        def get_cosine_score(self, s1, s2):
            s1_norm = s1 / torch.norm(s1, dim=1, keepdim=True)
            s2_norm = s2 / torch.norm(s2, dim=1, keepdim=True)
            cosine_score_ = (s1_norm * s2_norm).sum(dim=1)
            return cosine_score_
        """
    # 计算皮尔逊相关系数
    """
    def compute_correlation_coefficient(predictions_, labels_):
        predictions_, labels_ = np.array(predictions_), np.array(labels_)
        return np.corrcoef(predictions_, labels_)[0, 1]
    """


# 验证函数
def eval_metrics(model_, loader_):
    model_.eval()
    loss_sum, correlation_coefficient_sum, count = 0, 0, 0

    with no_grad():
        for s_, labels_ in loader_:
            s_, labels_ = s_.to(DEVICE), labels_.to(DEVICE)[::2]

            output_, cosine_score_ = model_(s_)

            loss_ = criterion(output_, labels_)
            loss_sum += loss_.item()

            # correlation_coefficient_sum += compute_correlation_coefficient(cosine_score_.detach().cpu().numpy(), labels_.detach().cpu().numpy())
            correlation_coefficient_sum += pearsonr(cosine_score_.detach().cpu().numpy(), labels_.detach().cpu().numpy())

            count += 1


    return loss_sum / count, correlation_coefficient_sum / count

# 数据加载
data_csv = pd.read_csv(path_to_csv_dataset)
data = convert_util(data_csv,"csv",3)
train, val = split_train_test(data)
train_loader = DataLoader(ATECDataset(train), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(ATECDataset(val), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)



# 模型、损失函数和优化器
model = SentenceBert().to(DEVICE)
criterion = nn.CrossEntropyLoss(reduction="mean")
optimizer = optim.AdamW(model.parameters(), lr=0.00003)
early_stop = EarlyStopping(patience=1, delta=0.01, path_bert=PATH_TO_BERT_MODEL, path_whole=PATH_TO_WHOLE_MODEL)


# 训练
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
            correlation_coefficient = pearsonr(cosine_score.detach().cpu().numpy(), labels.detach().detach().cpu().numpy())
            print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item()}, correlation_coefficient: {correlation_coefficient}")

    loss_val, correlation_coefficient_val = eval_metrics(model, val_loader)
    if early_stop(correlation_coefficient_val, model):
        print("Early stopped")
        break

    print(f"Validation - Epoch {epoch+1}, Loss: {loss_val}, correlation_coefficient: {correlation_coefficient_val}")

# 保存模型
save_checkpoint_torch(model, PATH_TO_WHOLE_MODEL, PATH_TO_BERT_MODEL)

