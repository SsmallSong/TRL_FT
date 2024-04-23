import pandas as pd

# 指定 Parquet 文件的路径
parquet_file = '/data2/huatong/imdb/plain_text/test-00000-of-00001.parquet'

# 使用 pandas 读取 Parquet 文件
df = pd.read_parquet(parquet_file)

# 查看数据集的前几行
print(df.head())