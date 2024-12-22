import pandas as pd
dff = pd.DataFrame()

sheets = pd.read_excel("./data/data.xlsx", sheet_name=None)
# 读取每个 sheet 中的句子数据
i = 0
for sheet_name, df in sheets.items():
    sentences = df['sentence'].tolist()  # 获取当前类的所有句子
    dff = pd.concat(
        [
            dff,
            pd.DataFrame({"sentence":sentences,"class":[i]*len(sentences)})
            ],
        ignore_index=True
    )
    i+=1

dff.to_csv("./data/dataset.csv", index=False)