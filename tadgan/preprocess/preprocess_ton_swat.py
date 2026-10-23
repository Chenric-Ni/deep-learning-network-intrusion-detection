import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# 读取数据文件（假设为 CSV 格式）
data = pd.read_csv(r'./dataset/TON_IoT_windows10/Train_Test_Windows_10.csv')

# 删除所有值为 0 的列
data_values = data.iloc[1:,:-1].replace("",0).fillna(0)  # 从第二行开始选择数据，将空值替换为 0
data_values = data_values.apply(pd.to_numeric, errors='coerce').fillna(0)
# 找到全为 0 的列
columns_to_drop = data_values.columns[(data_values == 0).all()]

# 删除这些列
data_cleaned = data.drop(columns=columns_to_drop)

# 提取异常标签（倒数第二列）和具体异常情况（倒数第一列）
labels = data_cleaned.iloc[:, -2]  # 异常标签（倒数第二列）
specific_labels = data_cleaned.iloc[:, -1]  # 具体异常情况列（倒数第一列）

# 删除标签列和具体异常情况列
data_cleaned = data_cleaned.drop(columns=[data_cleaned.columns[-1]])
data_cleaned = data_cleaned.drop(columns=[data_cleaned.columns[-1]])

# 划分训练集和测试集
train_data, test_data, train_labels, test_labels = train_test_split(
    data_cleaned, labels, test_size=0.3, random_state=42
)

# 保存为 train_data, train_labels, test_data, test_labels
train_data.to_csv(r'./dataset/TON_IoT_windows10/train_data.csv', index=False)
train_labels.to_csv(r'./dataset/TON_IoT_windows10/train_labels.csv', index=False)
test_data.to_csv(r'./dataset/TON_IoT_windows10/test_data.csv', index=False)
test_labels.to_csv(r'./dataset/TON_IoT_windows10/test_labels.csv', index=False)

# 输出信息
print("Train and test data saved successfully.")
print(f"Train data shape: {train_data.shape}")
print(f"Train labels shape: {train_labels.shape}")
print(f"Test data shape: {test_data.shape}")
print(f"Test labels shape: {test_labels.shape}")

