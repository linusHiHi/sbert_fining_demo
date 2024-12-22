from util.util import save_checkpoint_torch


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
            save_checkpoint_torch(model)
        elif corrcoef < self.best_loss - self.delta:
            self.best_loss = corrcoef
            save_checkpoint_torch(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


