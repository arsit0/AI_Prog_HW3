import os
import pandas as pd
import numpy as np
import matplotlib

# 强制用非交互后端
# 这样只保存图片，不会再弹 Figure 窗口
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.font_manager import FontProperties

# 直接指定 Windows 里的中文字体文件
# 不再让 matplotlib 自己去猜，不然很容易退回 DejaVu Sans
font_path = r"C:\Windows\Fonts\msyh.ttc"
if not os.path.exists(font_path):
    font_path = r"C:\Windows\Fonts\simhei.ttf"

myfont = FontProperties(fname=font_path)
plt.rcParams['axes.unicode_minus'] = False

# 题面写的是制表符分隔
# 但你这份实际数据已经验证过是逗号分隔
df = pd.read_csv('ICData.csv', encoding='utf-8-sig')

# 先看前5行
print("数据集前5行：")
print(df.head())

# 基本信息
print("\n数据集基本信息：")
print("行数：", df.shape[0])
print("列数：", df.shape[1])

print("\n各列数据类型：")
print(df.dtypes)

# 时间字段带秒，所以这里按秒解析
df['交易时间'] = pd.to_datetime(df['交易时间'], format='%Y/%m/%d %H:%M:%S')

# 提取小时
df['hour'] = df['交易时间'].dt.hour

# 构造搭乘站点数
df['ride_stops'] = (df['下车站点'] - df['上车站点']).abs()

# ride_stops=0 算异常，删掉
before_rows = df.shape[0]
df = df[df['ride_stops'] != 0].copy()
after_rows = df.shape[0]

print("\n删除 ride_stops = 0 的异常记录数：", before_rows - after_rows)

# 检查缺失值
print("\n各列缺失值数量：")
missing_counts = df.isnull().sum()
print(missing_counts)

# 有缺失值的话就删
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

# =========================
# 任务2(a) 早晚时段刷卡量统计
# =========================

# 只统计刷卡类型=0
df_bus = df[df['刷卡类型'] == 0].copy()

# 这里按题目要求必须用 numpy
hour_arr = df_bus['hour'].to_numpy()

early_count = np.sum(hour_arr < 7)
late_count = np.sum(hour_arr >= 22)
total_count = hour_arr.shape[0]

early_ratio = early_count / total_count * 100
late_ratio = late_count / total_count * 100

print("\n任务2(a) 早晚时段刷卡量统计：")
print("早峰前时段(hour < 7)刷卡量：", early_count)
print("占全天总刷卡量百分比：{:.2f}%".format(early_ratio))
print("深夜时段(hour >= 22)刷卡量：", late_count)
print("占全天总刷卡量百分比：{:.2f}%".format(late_ratio))

# =========================
# 任务2(b) 24小时柱状图
# =========================

# 统计24小时刷卡量，没有的小时补0
hour_counts = df_bus.groupby('hour').size().reindex(range(24), fill_value=0)

# 给不同时间段单独上色
colors = []
for h in range(24):
    if h < 7:
        colors.append('orange')
    elif h >= 22:
        colors.append('red')
    else:
        colors.append('skyblue')

plt.figure(figsize=(12, 6))
plt.bar(hour_counts.index, hour_counts.values, color=colors)

# 标题和坐标轴
plt.title('24小时刷卡量分布', fontproperties=myfont)
plt.xlabel('小时', fontproperties=myfont)
plt.ylabel('刷卡量（次）', fontproperties=myfont)

# x轴标签步长为2
plt.xticks(range(0, 24, 2))

# 水平网格线
plt.grid(axis='y', linestyle='--', alpha=0.5)

# 图例
legend_handles = [
    Patch(facecolor='orange', label='早峰前时段(<7)'),
    Patch(facecolor='red', label='深夜时段(>=22)'),
    Patch(facecolor='skyblue', label='其余时段')
]
plt.legend(handles=legend_handles, prop=myfont)

# 保存图片
plt.tight_layout()
plt.savefig('hour_distribution.png', dpi=150)
plt.close()

print("任务2(b)完成，图像已保存为 hour_distribution.png")