import pandas as pd

# 题面写的是制表符分隔
# 但实际跑出来发现整行都被读到一列里了
# 说明这份真实数据文件其实是逗号分隔
df = pd.read_csv('ICData.csv', encoding='utf-8-sig')

# 先看看前5行
print("数据集前5行：")
print(df.head())

# 打印基本信息
print("\n数据集基本信息：")
print("行数：", df.shape[0])
print("列数：", df.shape[1])

print("\n各列数据类型：")
print(df.dtypes)

# 实际数据里的时间带秒，所以这里按秒来解析
df['交易时间'] = pd.to_datetime(df['交易时间'], format='%Y/%m/%d %H:%M:%S')

# 提取小时
df['hour'] = df['交易时间'].dt.hour

# 计算搭乘站点数
df['ride_stops'] = (df['下车站点'] - df['上车站点']).abs()

# ride_stops 为0算异常记录，删掉
before_rows = df.shape[0]
df = df[df['ride_stops'] != 0].copy()
after_rows = df.shape[0]

print("\n删除 ride_stops = 0 的异常记录数：", before_rows - after_rows)

# 缺失值检查
print("\n各列缺失值数量：")
missing_counts = df.isnull().sum()
print(missing_counts)

# 如果有缺失值，这里先按删除处理
if missing_counts.sum() > 0:
    print("\n处理策略：删除含缺失值的记录")
    before_dropna = df.shape[0]
    df = df.dropna().copy()
    after_dropna = df.shape[0]
    print("删除缺失值记录数：", before_dropna - after_dropna)
else:
    print("\n未发现缺失值，无需处理")

print("\n预处理后数据行数：", df.shape[0])
print("任务1完成")