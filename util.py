import pandas as pd
import itertools


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
                        data.append([sentence1, sentence2, 0])

    return data


import itertools
import random

import itertools
import random

def convert_excel_to_classification_format_(sheets, negative_sample_ratio=0.1):
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
    for class_name, sentences in sentences_by_class.items():
        for other_class_name, other_sentences in sentences_by_class.items():
            if class_name != other_class_name:
                # 选择性地采样负样本
                num_negative_samples = int(len(other_sentences) * negative_sample_ratio)
                sampled_other_sentences = random.sample(other_sentences, num_negative_samples)

                # 创建负样本的句子对
                for sentence1 in sentences:
                    for sentence2 in sampled_other_sentences:
                        data.append([sentence1, sentence2, 0])

    return data



import torch


def save_checkpoint(model):
    """保存最好的模型状态"""
    torch.save(model.state_dict(), "./miaModel/whole.bin")
    torch.save(model.pre_train.state_dict(), "./miaModel/my_bert.bin")


class EarlyStopping:
    def __init__(self, patience=3, delta=0):
        """
        :param patience: 当验证集损失在多少个epoch内没有改善时，停止训练
        :param delta: 假设验证集损失在变化时，才认为是改善（防止微小的波动导致早停）
        """
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_loss = None

    def __call__(self, corrcoef, model):
        """
        :param corrcoef: 当前验证集的损失
        :param model: 当前模型（如果早停则保存最好的模型）
        :return: 如果满足早停条件返回True，否则返回False
        """
        if self.best_loss is None:
            self.best_loss = corrcoef
            save_checkpoint(model)
        elif corrcoef < self.best_loss - self.delta:
            self.best_loss = corrcoef
            save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

from sklearn.model_selection import train_test_split

def split_data(data, train_size=0.7, val_size=0.15, test_size=0.15, random_state=None):
    # 首先，划分训练集和剩余集（验证集 + 测试集）
    train_data, temp_data = train_test_split(data, train_size=train_size, random_state=random_state)

    # 然后，将剩余集划分为验证集和测试集
    val_data, test_data = train_test_split(temp_data, train_size=val_size / (val_size + test_size), random_state=random_state)

    return train_data, val_data, test_data
