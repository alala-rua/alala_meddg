import pickle
import os

all_data_path = "./data/new_train.pk"
train_data_path = "./data/train_data.pk"
dev_data_path = "./data/dev_data.pk"


with open(all_data_path, "rb") as f:
    all_data = pickle.load(f)


from sklearn.model_selection import train_test_split

train_data, dev_data = train_test_split(all_data, test_size=0.1, random_state=123456)

with open(train_data_path, "wb") as f:
    pickle.dump(train_data, f)

with open(dev_data_path, "wb") as f:
    pickle.dump(dev_data, f)

print("数据预处理完成")