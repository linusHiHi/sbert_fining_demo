import torch

def save_checkpoint_torch(model,path_to_whole_model, path_to_bert_model):
    """保存最好的模型状态"""
    torch.save(model.state_dict(), path_to_whole_model)
    torch.save(model.pre_train.state_dict(), path_to_bert_model)




