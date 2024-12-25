import pandas as pd

whole = pd.concat(
    [
        pd.read_csv("../source_data.csv"),
        pd.read_csv("../found_class_0_3.csv"),
        # pd.read_csv("../__(ready_to_del)__found_class_0_to_2.csv"),

    ]
)
whole.to_csv("../full_data_with_all_new_sentence.csv", index=False)