import pandas as pd

whole = pd.concat(
    [
        pd.read_csv("./data/dataset.csv"),
        pd.read_csv("./data/error.csv"),
        pd.read_csv("./data/error.csv"),

    ]
)
whole.to_csv("./data/dataset_whole.csv", index=False)