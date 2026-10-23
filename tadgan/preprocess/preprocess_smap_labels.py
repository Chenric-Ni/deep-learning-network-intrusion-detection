import numpy as np
import pandas as pd
import ast
import os

# 1. 加载 CSV 文件
df = pd.read_csv('./dataset/SMAP/labeled_anomalies.csv')

# 遍历 CSV 中的每一行，每一行对应一台机器
for idx, row in df.iterrows():
    chan_id = row['chan_id']
    anomaly_sequence_str = row['anomaly_sequences']  # 注意字段名称是否与CSV中一致
    num_values = int(row['num_values'])
    
    # 2. 将 anomaly_sequence 字符串转换为 Python 列表
    try:
        anomaly_intervals = ast.literal_eval(anomaly_sequence_str)
    except Exception as e:
        print(f"解析 chan_id {chan_id} 的 anomaly_sequence 出错: {e}")
        continue

    # 3. 初始化标签数组，长度为 num_values，全部初始化为 0（表示正常）
    labels = np.zeros(num_values, dtype=int)
    
    # 4. 遍历每个异常区间，将对应区间标记为 1（表示异常）
    for interval in anomaly_intervals:
        if isinstance(interval, list) and len(interval) == 2:
            start, end = interval
            # 这里假设异常区间为闭区间，即[start, end]均为异常，如果是半闭区间请调整
            labels[start:end+1] = 1
        else:
            print(f"chan_id {chan_id} 中的异常区间格式异常: {interval}")
    
    # 5. 保存为 .npy 文件，文件名可以包含 chan_id
    out_put_dir = './dataset/SMAP/labels'
    output_filename = os.path.join(out_put_dir, f"{chan_id}_labels.npy")
    np.save(output_filename, labels)
    print(f"chan_id {chan_id} 的标签已保存为 {output_filename}")
