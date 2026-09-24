# %%
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, \
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_curve, auc, classification_report

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
from math import sqrt

plt.rcParams['font.sans-serif'] = ['simhei']
plt.rcParams['font.serif'] = ['simhei']
plt.rcParams['axes.unicode_minus'] = False  # 默认是使用Unicode负号，设置正常显示字符，如正常显示负号

import seaborn as sns

sns.set_style("darkgrid", {"font.sans-serif": ['simhei', 'Droid Sans Fallback']})
import pandas as pd

# %%
data = pd.read_csv('A题-全部数据/附件2/M202.csv')
data.head()

# %%
data.shape

# %%
X = data[['时间', '物料推送气缸推送状态', '物料推送气缸收回状态', '物料推送数', '物料待抓取数',
          '放置容器数', '容器上传检测数', '填装检测数', '填装定位器固定状态', '填装定位器放开状态', '物料抓取数',
          '填装旋转数',
          '填装下降数', '填装数', '加盖检测数', '加盖定位数', '推盖数', '加盖下降数', '加盖数', '拧盖检测数',
          '拧盖定位数', '拧盖下降数', '拧盖旋转数', '拧盖数', '合格数', '不合格数']]

# %%
import joblib

# load the saved model
model = joblib.load('./modellist/Q1_物料推送装置故障1001.bin')

# predict using the loaded model
model.predict(X)

# %%
X = np.array(X)

# %%
import pandas as pd
import joblib

# 假设 X 已经是一个合适的输入数据格式
# X = ...

# 加载模型
models = {
    "1001": joblib.load('./modellist/Q1_物料推送装置故障1001.bin'),
    "2001": joblib.load('./modellist/Q1_物料检测装置故障2001.bin'),
    "4003": joblib.load('./modellist/Q1_填装装置填装故障4003.bin'),
    "4001": joblib.load('./modellist/Q1_填装装置检测故障4001.bin'),
    "4002": joblib.load('./modellist/Q1_填装装置定位故障4002.bin'),
    "6002": joblib.load('./modellist/Q1_拧盖装置拧盖故障6002.bin'),
    "5002": joblib.load('./modellist/Q1_加盖装置加盖故障5002.bin'),
    "5001": joblib.load('./modellist/Q1_加盖装置定位故障5001.bin'),
    "6001": joblib.load('./modellist/Q1_拧盖装置定位故障6001.bin')
}

# 对新的输入数据 X 进行预测，并收集预测结果
predictions = {}
for code, model in models.items():
    predictions[code] = model.predict(X)

# %%
for i, j in predictions.items():
    data[i] = j
    print(i)
    print(data[i].value_counts())

# %%
a = data[data['1001'] == 1]

# %%
# 根据用户的明确说明，连续的时间值表示持续的故障事件，
# 我们应该相应地将这些事件分组来计算持续时间。

# 提取唯一的日期，以便分别处理每天的数据
days = a['日期'].unique()

# 初始化一个空的DataFrame，用于存储处理后的结果
processed_data = pd.DataFrame(columns=['序号', '日期', '开始时间', '持续时长/秒'])

# 对每一天的数据进行处理，计算连续事件的持续时间
for day in days:
    # 筛选出当前天的数据，并按时间排序
    day_data = a[a['日期'] == day].sort_values(by='时间')

    # 初始化变量，用于跟踪连续事件
    start_time = None  # 开始时间
    last_time = None  # 上一个时间点
    duration = 0  # 持续时间

    # 用于存储当天事件的列表
    events = []

    for index, row in day_data.iterrows():
        if start_time is None:
            # 开始一个新的事件
            start_time = last_time = row['时间']
            duration = 1
        elif row['时间'] == last_time + 1:
            # 继续当前的事件
            last_time = row['时间']
            duration += 1
        else:
            # 结束当前事件并开始一个新的事件
            events.append((day, start_time, duration))
            start_time = last_time = row['时间']
            duration = 1

    # 不要忘记添加最后一个事件
    if start_time is not None:
        events.append((day, start_time, duration))

    # 将事件转换为DataFrame并追加到处理后的数据中
    day_events_df = pd.DataFrame(events, columns=['日期', '开始时间', '持续时长/秒'])
    day_events_df['序号'] = range(1, len(day_events_df) + 1)
    processed_data = pd.concat([processed_data, day_events_df], ignore_index=True)
    del processed_data['序号']

processed_data

# %%

# 统计每个月的故障总次数、最长和最短的持续时长
monthly_stats = processed_data.groupby('日期').agg(
    故障总次数=('开始时间', 'count'),
    最长持续时长=('持续时长/秒', 'max'),
    最短持续时长=('持续时长/秒', 'min')
)

monthly_stats

# %%
shapemax = 0
temp = pd.DataFrame()
for d in ['1001', '2001', '4001', '4002', '4003', '5001', '5002', '6001', '6002']:
    a = data[data[d] == 1]
    # 根据用户的明确说明，连续的时间值表示持续的故障事件，
    # 我们应该相应地将这些事件分组来计算持续时间。

    # 提取唯一的日期，以便分别处理每天的数据
    days = a['日期'].unique()

    # 初始化一个空的DataFrame，用于存储处理后的结果
    processed_data = pd.DataFrame(columns=['序号', '日期', '开始时间', '持续时长/秒'])

    # 对每一天的数据进行处理，计算连续事件的持续时间
    for day in days:
        # 筛选出当前天的数据，并按时间排序
        day_data = a[a['日期'] == day].sort_values(by='时间')

        # 初始化变量，用于跟踪连续事件
        start_time = None  # 开始时间
        last_time = None  # 上一个时间点
        duration = 0  # 持续时间

        # 用于存储当天事件的列表
        events = []

        for index, row in day_data.iterrows():
            if start_time is None:
                # 开始一个新的事件
                start_time = last_time = row['时间']
                duration = 1
            elif row['时间'] == last_time + 1:
                # 继续当前的事件
                last_time = row['时间']
                duration += 1
            else:
                # 结束当前事件并开始一个新的事件
                events.append((day, start_time, duration))
                start_time = last_time = row['时间']
                duration = 1

        # 不要忘记添加最后一个事件
        if start_time is not None:
            events.append((day, start_time, duration))

        # 将事件转换为DataFrame并追加到处理后的数据中
        day_events_df = pd.DataFrame(events, columns=['日期', '开始时间', '持续时长/秒'])
        day_events_df['序号'] = range(1, len(day_events_df) + 1)
        processed_data = pd.concat([processed_data, day_events_df], ignore_index=True)
    if shapemax < processed_data.shape[0]:
        shapemax = processed_data.shape[0]
    del processed_data['序号']
    print(temp.shape)
    temp = pd.concat([temp, processed_data], axis=1)
    # 统计每个月的故障总次数、最长和最短的持续时长
    monthly_stats = processed_data.groupby('日期').agg(
        故障总次数=('开始时间', 'count'),
        最长持续时长=('持续时长/秒', 'max'),
        最短持续时长=('持续时长/秒', 'min')
    )
    print(d)
    display(monthly_stats)

# %%
# 新建一列名为“示例列”，并随机赋予一些值
temp.insert(0, '序号', range(1, shapemax + 1))
temp

# %%
temp.to_excel('result2_M202.xlsx', index=None)

# %%


# %%



