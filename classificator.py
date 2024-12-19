import torch
import torch.nn as nn
import torch.optim as optim


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 假设 SBERT 的嵌入维度是 384
input_dim = 384
hidden_dim = 128
output_dim = 3

# 初始化分类器
classifier = Classifier(input_dim, hidden_dim, output_dim)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

# 假设你有训练数据的嵌入和标签
train_embeddings = torch.tensor(sentence_embeddings)  # 训练数据的句子嵌入
train_labels = torch.tensor([0, 1])  # 对应的标签

# 训练分类器
for epoch in range(10):  # 迭代训练10轮
    classifier.train()
    optimizer.zero_grad()

    # 前向传播
    outputs = classifier(train_embeddings)

    # 计算损失
    loss = criterion(outputs, train_labels)

    # 反向传播和优化
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

torch.save(classifier.state_dict(), "./model/classifier.bin")

""" usage
model = Classifier(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load("./model/classifier.bin"))
model.eval()  # 设置模型为评估模式

with torch.no_grad():
    outputs = model(sample_input)
    probabilities = F.softmax(outputs, dim=1)  # 使用softmax转换为概率分布
    predicted_classes = torch.argmax(probabilities, dim=1)  # 获取预测的类别

print(f"Predicted classes: {predicted_classes}")


"""

