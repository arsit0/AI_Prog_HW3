import os
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns

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
# =========================
# 任务3 线路站点分析
# =========================

def analyze_route_stops(df, route_col='线路号', stops_col='ride_stops'):
    """
    计算各线路乘客的平均搭乘站点数及其标准差。
    Parameters
    ----------
    df : pd.DataFrame  预处理后的数据集
    route_col : str    线路号列名
    stops_col : str    搭乘站点数列名
    Returns
    -------
    pd.DataFrame  包含列：线路号、mean_stops、std_stops，按 mean_stops 降序排列
    """
    result = df.groupby(route_col)[stops_col].agg(['mean','std']).reset_index()
    result.columns = ['线路号','mean_stops','std_stops']
    result = result.sort_values(by='mean_stops', ascending=False).reset_index(drop=True)
    return result

# 先调用函数
route_stats = analyze_route_stops(df)

# 打印前10行
print("\n任务3 各线路平均搭乘站点数及其标准差（前10行）：")
print(route_stats.head(10))

# 取均值最高的前15条线路
top15_routes = route_stats.head(15).copy()

# 为了让图里从上到下是高到低，这里倒一下
plot_data = top15_routes.iloc[::-1].copy()

# seaborn 画分类轴时，转成字符串更稳一点
plot_data['线路号_str'] = plot_data['线路号'].astype(str)

plt.figure(figsize=(12,8))

# 这里把 hue 也设成线路号本身
# 这样 palette='Blues_d' 就不会再报 FutureWarning
ax = sns.barplot(
    data=plot_data,
    x='mean_stops',
    y='线路号_str',
    hue='线路号_str',
    orient='h',
    palette='Blues_d',
    dodge=False,
    legend=False
)

# 标准差误差棒补上
ax.errorbar(
    x=plot_data['mean_stops'],
    y=np.arange(len(plot_data)),
    xerr=plot_data['std_stops'],
    fmt='none',
    ecolor='black',
    elinewidth=1,
    capsize=0.3
)

plt.title('均值最高前15条线路的平均搭乘站点数', fontproperties=myfont)
plt.xlabel('平均搭乘站点数', fontproperties=myfont)
plt.ylabel('线路号', fontproperties=myfont)

# x轴从0开始
plt.xlim(left=0)

# 不再手动 set_yticklabels 了
# 直接把现有刻度标签逐个设字体，这样不会报 set_ticklabels 的 warning
for label in ax.get_yticklabels():
    label.set_fontproperties(myfont)

for label in ax.get_xticklabels():
    label.set_fontproperties(myfont)

plt.tight_layout()
plt.savefig('route_stops.png', dpi=150)
plt.close()

print("任务3完成，图像已保存为 route_stops.png")
# =========================
# 任务4 高峰小时系数 PHF
# =========================

# 这里继续用刷卡类型=0的数据
# 因为前面任务2统计上车刷卡量就是按这个口径来的
# 题目这里说的是刷卡量，沿用这个口径更一致
hourly_counts = df_bus.groupby('hour').size().reindex(range(24), fill_value=0)

# 自动找出刷卡量最大的那个小时
peak_hour = hourly_counts.idxmax()
peak_hour_count = hourly_counts.max()

# 只取高峰小时内的记录
peak_df = df_bus[df_bus['hour'] == peak_hour].copy()

# 下面要做分钟粒度聚合
# floor('5min') 的意思是把时间往下归到所在的5分钟起点
# 比如 09:17:36 会归到 09:15:00
peak_df['time_5min'] = peak_df['交易时间'].dt.floor('5min')
peak_df['time_15min'] = peak_df['交易时间'].dt.floor('15min')

# 统计高峰小时内每个5分钟窗口的刷卡量
count_5min = peak_df.groupby('time_5min').size().sort_index()
max_5min_time = count_5min.idxmax()
max_5min_count = count_5min.max()

# PHF5 按题目公式直接算
phf5 = peak_hour_count / (12 * max_5min_count)

# 统计高峰小时内每个15分钟窗口的刷卡量
count_15min = peak_df.groupby('time_15min').size().sort_index()
max_15min_time = count_15min.idxmax()
max_15min_count = count_15min.max()

# PHF15 按题目公式直接算
phf15 = peak_hour_count / (4 * max_15min_count)

# 下面只是把时间格式整理成题目想要的样子
peak_hour_start = f"{peak_hour:02d}:00"
peak_hour_end = f"{(peak_hour + 1) % 24:02d}:00"

max_5min_start = max_5min_time.strftime('%H:%M')
max_5min_end = (max_5min_time + pd.Timedelta(minutes=5)).strftime('%H:%M')

max_15min_start = max_15min_time.strftime('%H:%M')
max_15min_end = (max_15min_time + pd.Timedelta(minutes=15)).strftime('%H:%M')

print("\n任务4 高峰小时系数计算结果：")
print(f"高峰小时：{peak_hour_start} ~ {peak_hour_end}，刷卡量：{peak_hour_count} 次")
print(f"最大5分钟刷卡量（{max_5min_start}~{max_5min_end}）：{max_5min_count} 次")
print(f"PHF5  = {peak_hour_count} / (12 × {max_5min_count}) = {phf5:.4f}")
print(f"最大15分钟刷卡量（{max_15min_start}~{max_15min_end}）：{max_15min_count} 次")
print(f"PHF15 = {peak_hour_count} / ( 4 × {max_15min_count}) = {phf15:.4f}")
# =========================
# 任务5 线路驾驶员信息批量导出
# =========================

# 先筛出 1101 到 1120 这20条线路
route_driver_df = df[(df['线路号'] >= 1101) & (df['线路号'] <= 1120)].copy()

# 在程序根目录创建文件夹
output_dir = '线路驾驶员信息'
os.makedirs(output_dir, exist_ok=True)

print("\n任务5 生成的文件路径：")

# 按题目要求，固定输出20个文件
for route in range(1101, 1121):
    one_route = route_driver_df[route_driver_df['线路号'] == route].copy()

    # 只保留 车辆编号 和 驾驶员编号
    # 去重后再按车辆编号、驾驶员编号排一下，写出来更整齐
    pair_df = one_route[['车辆编号', '驾驶员编号']].drop_duplicates().sort_values(
        by=['车辆编号', '驾驶员编号']
    )

    # 驾驶员编号当前是 float，导出时转成整数样子，不写成 41.0 这种
    if not pair_df.empty:
        pair_df['驾驶员编号'] = pair_df['驾驶员编号'].astype(int)

    file_path = os.path.join(output_dir, f'{route}.txt')

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"线路号: {route}\n")
        f.write("车辆编号\t驾驶员编号\n")

        for _, row in pair_df.iterrows():
            f.write(f"{int(row['车辆编号'])}\t\t{int(row['驾驶员编号'])}\n")

    print(file_path)

print("任务5完成，20个线路文件已输出成功")