# sbert_fining_demo
## intro
参考一些网络博客后，我们写了一个基于sentence-transformers的微调脚本，对[bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)进行微调。


## 项目结构
### 主要脚本
- 微调脚本
 -   - fine_tune_with_pytorch.py: 被放弃的方向，作用已然不是很大。输出目录在**./pytorch_result/**
 -   - fine_tune_with_sbert_api.py: 效果还可以。模型输出目录在**./sbert_result/**
- 测试脚本
-  - 只写了针对 **fine_tune_with_sbert_api.py**的脚本，
-  - 输出被错误识别为正常类的**其他类**句子，目标在[found_class_3.csv](/home/mia/Documents/python/ai/sbert_fining_demo/data/found_class_3.csv)
### 数据集
`
data  
├── source_data.xlsx        原始的excel表格  
├── source_data.csv         原始excel转格式   
├── found_class_0_to_2.csv  测试中不理想的问题（应该为0-2类但是最大相似度太低）  
├── found_class_3.csv       测试中不理想的问题（应该为3*（其他）*类但是最大相似度太高  
├── new_class_0_train.csv   把xx替换为地名后的文件  
*****************
├── full_data_with_all_new_sentence.csv 将所有的文件统合后的
├── sampled_data_with_all_new_sentence.csv针对class 0 规模太大（不平衡），进行了采样  
*****************
├── script      用来数据处理的脚本  
│ ├── concat_all.py  
│ ├── exel_to_csv_adding_new_class_0.py  
│ ├── insert_new_class_0_to_2.py  
│ └── sampling_class_0_in_full_data_.py  
└── raw  
  └── raw_test
`
### bert-base-chinese
原始的sbert模型
### pytorch_result and sbert_result 
微调后的模型
### script and openmax
存放了一些未写完或者被丢弃的代码