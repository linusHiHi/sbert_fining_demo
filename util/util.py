
import itertools
from sklearn.model_selection import train_test_split
import torch
from sentence_transformers import InputExample


def convert_excel_to_classification_format(sheets):
    # 读取 Excel 文件中的所有工作表
    # 用于存储结果
    data = []

    # 用于存储所有的句子对
    sentences_by_class = {}

    # 读取每个 sheet 中的句子数据
    for sheet_name, df in sheets.items():
        sentences = df['sentence'].tolist()  # 获取当前类的所有句子
        sentences_by_class[sheet_name] = sentences

    # 处理每个类的句子
    for class_name, sentences in sentences_by_class.items():
        # 将相同类中的句子两两组合，并添加标签 1
        for sentence1, sentence2 in itertools.combinations(sentences, 2):
            data.append([sentence1, sentence2, 1])

        # 处理不同类之间的句子对
        for other_class_name, other_sentences in sentences_by_class.items():
            if other_class_name != class_name:
                for sentence1 in sentences:
                    for sentence2 in other_sentences:
                        if [sentence1, sentence2, 0] not in data and [sentence2, sentence1, 0] not in data:
                            data.append([sentence1, sentence2, 0])

    return list(data)





def save_checkpoint_torch(model,path_to_whole_model, path_to_bert_model):
    """保存最好的模型状态"""
    torch.save(model.state_dict(), path_to_whole_model)
    torch.save(model.pre_train.state_dict(), path_to_bert_model)




def split_data(data, train_size=0.7, val_size=0.15, test_size=0.15, random_state=None):
    # 首先，划分训练集和剩余集（验证集 + 测试集）
    train_data, temp_data = train_test_split(data, train_size=train_size, random_state=random_state)

    # 然后，将剩余集划分为验证集和测试集
    val_data, test_data = train_test_split(temp_data, train_size=val_size / (val_size + test_size), random_state=random_state)

    return train_data, val_data, test_data

def sampling(datas):
  samples = []
  for item in datas:
    samples.append(InputExample(texts=[item[0], item[1]], label=float(item[2])))
  return samples