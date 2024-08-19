# import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_json('dataset/VQA_RAD_Dataset_Public.json')

df["answer_count"] = df["answer"].apply(lambda x: len(str(x).split()))
# length_distribution = df[df["answer_type"]=="CLOSED"]["answer_count"].value_counts().sort_index()
length_distribution = df[df["answer_type"]=="OPEN"]["answer_count"].value_counts().sort_index()
# length_distribution = df["answer_count"].value_counts().sort_index()

print(set(df[df["answer_type"] == "CLOSED"]["answer"]))


# Plotting the distribution
# plt.figure(figsize=(10, 6))
# ax = sns.histplot(df[df["answer_type"]=="CLOSED"]["answer_count"], kde=False)

# # Annotating the bars with the counts
# for p in ax.patches:
#     height = p.get_height()
#     if height > 0:  # Only annotate non-zero bars
#         plt.text(p.get_x() + p.get_width() / 2., height + 0.5, int(height), ha="center", fontsize=10)

# plt.title('Distribution of Answer Lengths')
# plt.xlabel('Answer Length')
# plt.ylabel('Frequency')
# # plt.show()
# plt.savefig("closed.png")


# open_question = df[df["answer_type"]=="CLOSED"]
# open_question["answer_count"] = open_question["answer"].apply(lambda x: len(str(x).split()))
# print(open_question["answer_count"])

# # Count the distribution of answer lengths
# length_distribution = open_question["answer_count"].value_counts().sort_index()

# print(length_distribution)



# splits = {'train': 'data/train-00000-of-00001-eb8844602202be60.parquet', 'test': 'data/test-00000-of-00001-e5bc3d208bb4deeb.parquet'}
# df = pd.read_parquet("hf://datasets/flaviagiammarino/vqa-rad/" + splits["train"])

# a = 1
# print(df)
# print(len(df))
# print(df["answer_type"])
# json.load("VQA_RAD_Dataset_Public.json")