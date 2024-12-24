import pandas as pd

path_to_full_data = "../full_data_with_all_new_sentence.csv"
path_to_sampled_data = "../sampled_data_with_all_new_sentence.csv"
df = pd.read_csv(path_to_full_data)
df_class_0 = df[df["class"]==0]
df_class_other =df[df["class"]!=0]

N = min(int(len(df_class_other)/3), len(df_class_0))

df_class_0_new = df_class_0.sample(n=N)

df = pd.concat([df_class_0_new, df_class_other]).to_csv(path_to_sampled_data, index=False)