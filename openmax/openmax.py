import torch
import torch.nn.functional as F
from scipy.spatial.distance import euclidean
from scipy.stats import weibull_min

from fine_tuning import train_loader


# 计算每个类别的平均激活向量（MAV）
def compute_MAV(model_, data_loader, num_classes_):
    model_.eval()
    mavs_ = [torch.zeros(64) for _ in range(num_classes_)]
    counts = [0] * num_classes_

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.view(-1, 28 * 28)
            _, features = model_(images)
            for i, label in enumerate(labels):
                mavs_[label] += features[i]
                counts[label] += 1

    for i in range(num_classes_):
        mavs_[i] /= counts[i]

    return mavs_


# 计算每个类别的激活向量分布并拟合威布尔分布
def fit_weibull_distributions(model_, data_loader, mavs_, num_classes, tailsize=20):
    model_.eval()
    distances = [[] for _ in range(num_classes)]

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.view(-1, 28 * 28)
            _, features = model_(images)
            for i, label in enumerate(labels):
                distance = euclidean(features[i].numpy(), mavs_[label].numpy())
                (distances[label]).append(distance)

    weibull_parameters = []

    for i in range(num_classes):
        distances[i].sort()
        mr = distances[i][-tailsize:]
        weibull_model = weibull_min.fit(mr, floc=0)
        weibull_parameters.append(weibull_model)


    return weibull_parameters


# 计算 OpenMax 概率
def openmax_probability(model_, image, mavs_, weibull_models_, num_classes_, alpha=10):
    model_.eval()
    with torch.no_grad():
        image = image.view(-1, 28 * 28)
        logits, features = model_(image)
        logits = logits.squeeze()
        features = features.squeeze()

    openmax_scores = logits.clone()
    for i in range(num_classes_):
        dist = euclidean(features.numpy(), mavs_[i].numpy())
        weibull_model = weibull_models_[i]
        w_score = weibull_min.cdf(dist, *weibull_model)
        openmax_scores[i] = logits[i] * (1 - w_score)

    openmax_unknown = torch.sum(logits - openmax_scores)
    openmax_scores = torch.cat((openmax_scores, openmax_unknown.unsqueeze(0)))

    openmax_prob = F.softmax(openmax_scores, dim=0)
    return openmax_prob


# 计算 MAVs 和 Weibull 分布
num_classes = 3
model = torch.load('.pt')
mavs = compute_MAV(model, train_loader, num_classes)
weibull_models = fit_weibull_distributions(model, train_loader, mavs, num_classes)


model = torch.load("./miaModel/whole.bin")

torch.save(mavs, "miaModel/mavs.pth")
torch.save(weibull_models, "miaModel/weibull.pth")
"""
qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
"""

# 示例用法
# 假设您已经有了训练好的模型和数据加载器 train_loader


"""
num_classes = 3
model = torch.load("./miaModel/whole.bin")
# 对新样本计算 OpenMax 概率
new_thing = "这个热水袋是不是很热。"
openmax_prob = openmax_probability(model, new_thing, mavs, weibull_models, num_classes)
"""
