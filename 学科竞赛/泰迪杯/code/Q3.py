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
datadf = pd.read_csv('A题-全部数据/附件3/M301.csv')
datadf.head()

# %%
y_variables = [
    '物料推送装置故障1001', '物料检测装置故障2001', '填装装置检测故障4001', '填装装置定位故障4002',
    '填装装置填装故障4003', '加盖装置定位故障5001', '加盖装置加盖故障5002', '拧盖装置定位故障6001',
    '拧盖装置拧盖故障6002'
]

for k in y_variables:
    datadf[k] = datadf[k].apply(lambda x: 0 if x == 0 else 1)

# %% [markdown]
# 为了将生产线的运行记录数据汇总成一条数据，我们需要设计一些汇总变量，这些变量能够代表生产线在一年内的整体运行情况。以下是一些可能的汇总变量及其计算公式和原理：
#
# 1. **总运行时间**：计算生产线在一年内运行的总时间。
#    - 公式：`总运行时间 = ∑（结束时间 - 开始时间）`
#    - 原理：累加每个生产周期的运行时间。
#
# 2. **总生产数量**：计算一年内生产线生产的总产品数量。
#    - 公式：`总生产数量 = ∑合格数`
#    - 原理：累加每天记录的合格产品数量。
#
# 3. **总不合格数量**：计算一年内生产线生产的不合格产品总数量。
#    - 公式：`总不合格数量 = ∑不合格数`
#    - 原理：累加每天记录的不合格产品数量。
#
# 4. **平均生产效率**：计算生产线的平均生产效率，即每小时生产的合格产品数量。
#    - 公式：`平均生产效率 = 总生产数量 / 总运行时间`
#    - 原理：用总生产数量除以总运行时间得到每小时的生产效率。
#
# 5. **设备综合故障率**：计算所有设备故障的综合故障率。
#     - 公式：`设备综合故障率 = （物料推送装置故障1001 + 物料检测装置故障2001 + ... + 拧盖装置拧盖故障6002） / 总天数 * 100%`
#     - 原理：累加所有设备故障的发生次数，然后除以一年的总天数，得到设备综合故障率。
#
# 6. **物料推送效率**：计算物料推送气缸的平均推送效率。
#    - 公式：`物料推送效率 = （物料推送数 / (物料推送数 + 物料待抓取数)） * 100%`
#    - 原理：用成功推送的物料数量除以（成功推送的物料数量加上待抓取的物料数量），得到推送效率。
#
# 7. **填装效率**：计算填装过程的效率。
#    - 公式：`填装效率 = （填装数 / 物料推送数） * 100%`
#    - 原理：用完成填装的次数除以成功推送的物料数量，得到填装效率。
#
# 8. **加盖效率**：计算加盖过程的效率。
#    - 公式：`加盖效率 = （加盖数 / 填装数） * 100%`
#    - 原理：用完成加盖的次数除以完成填装的次数，得到加盖效率。
#
# 9. **拧盖效率**：计算拧盖过程的效率。
#    - 公式：`拧盖效率 = （拧盖数 / 加盖数） * 100%`
#    - 原理：用完成拧盖的次数除以完成加盖的次数，得到拧盖效率。
#
# 10. **生产总周期数**：计算生产线一年内的总生产周期数。
#     - 公式：`生产总周期数 = ∑填装数`
#     - 原理：累加每天记录的填装次数，得到总生产周期数。
#
# 11. **合格率**：计算生产线的合格率。
#     - 公式：`合格率 = （总生产数量 / (总生产数量 + 总不合格数量)） * 100%`
#     - 原理：用总生产数量除以（总生产数量加上总不合格数量），得到合格率。
#
# 这些汇总变量可以帮助我们从不同角度评估生产线的运行状况，包括生产效率、产品质量、设备可靠性等。在实际应用中，可以根据需要选择适当的汇总变量进行分析。

# %%
# 创建一个空的DataFrame用于存储结果
df = pd.DataFrame()
count = 0

for k in datadf['日期'].unique():
    # 计算总运行时间
    data = datadf[datadf['日期'] == k]
    total_operation_time = data.shape[0]

    # 计算总生产数量和总不合格数量
    total_production = data['合格数'].sum()
    total_defects = data['不合格数'].sum()

    # 计算平均生产效率
    average_efficiency = total_production / total_operation_time if total_operation_time else 0

    # 假设每条记录代表一个小时的运行，总天数为记录中的日期数
    # 计算每种故障每天的发生次数
    daily_faults = data.groupby('日期')[y_variables].sum()
    total_days = data['日期'].nunique()
    # 对每一天计算是否有任何故障发生
    any_faults_per_day = daily_faults.any(axis=1)

    # 计算有故障发生的天数
    unique_fault_days = any_faults_per_day.sum()

    # 计算修正后的设备综合故障率
    corrected_composite_fault_rate = (unique_fault_days / total_days) * 100 if total_days else 0

    # 计算物料推送效率
    material_push_efficiency = (data['物料推送数'].sum() / (
                data['物料推送数'].sum() + data['物料待抓取数'].sum())) * 100

    # 计算填装效率
    filling_efficiency = (data['填装检测数'].sum() / data['物料推送数'].sum()) * 100 if data['物料推送数'].sum() else 0

    # 计算加盖效率
    capping_efficiency = (data['加盖数'].sum() / data['填装检测数'].sum()) * 100 if data['填装检测数'].sum() else 0

    # 计算拧盖效率
    screwing_efficiency = (data['拧盖数'].sum() / data['加盖数'].sum()) * 100 if data['加盖数'].sum() else 0

    # 计算设备综合故障率
    # 修正设备综合故障率的计算

    # 提取所有设备故障的列名
    y_variables = data.columns[data.columns.str.contains('故障')]

    # 计算生产总周期数
    total_production_cycles = data['填装检测数'].sum()

    # 计算合格率
    acceptance_rate = (total_production / (total_production + total_defects) * 100) if (
                total_production + total_defects) else 0

    additional_data = {
        "日期": k,
        "总运行时间": total_operation_time,
        "总生产数量": total_production,
        "总不合格数量": total_defects,
        "平均生产效率": average_efficiency,
        "设备综合故障率": corrected_composite_fault_rate,
        "物料推送效率": material_push_efficiency,
        "填装效率": filling_efficiency,
        "加盖效率": capping_efficiency,
        "拧盖效率": screwing_efficiency,
        "生产总周期数": total_production_cycles,
        "合格率": acceptance_rate
    }

    df = pd.concat([df, pd.DataFrame(additional_data, index=[count])])
    count = count + 1

# %%
df

# %%
tt = pd.DataFrame({
    '操作人员编号': ['A001', 'A002', 'A003', 'A004', 'A005', 'A006', 'A007', 'A008', 'A009', 'A010'],
    '工龄': [1, 5, 6, 4, 2, 3, 4, 1, 2, 5],
    '生产线编号': ['M301', 'M302', 'M303', 'M304', 'M305', 'M306', 'M307', 'M308', 'M309', 'M310']
})
tt

# %%
df['操作人员编号'] = tt['操作人员编号'].iloc[0]
df['工龄'] = tt['工龄'].iloc[0]
df['生产线编号'] = tt['生产线编号'].iloc[0]

# %%
df

# %%
import pandas as pd
import numpy as np

# 定义文件名列表和Y变量列表
filenames = [f"A题-全部数据/附件3/M10{i}.csv" for i in range(1, 10)] + ["A题-全部数据/附件3/M110.csv"]
alldf = pd.DataFrame()
N = 0

# 遍历每个文件
for filename in filenames:
    # 读取CSV文件
    datadf = pd.read_csv(filename)
    for k in y_variables:
        datadf[k] = datadf[k].apply(lambda x: 0 if x == 0 else 1)
    # 创建一个空的DataFrame用于存储结果
    df = pd.DataFrame()
    count = 0

    for k in datadf['日期'].unique():
        # 计算总运行时间
        data = datadf[datadf['日期'] == k]
        total_operation_time = data.shape[0]

        # 计算总生产数量和总不合格数量
        total_production = data['合格数'].sum()
        total_defects = data['不合格数'].sum()

        # 计算平均生产效率
        average_efficiency = total_production / total_operation_time if total_operation_time else 0

        # 假设每条记录代表一个小时的运行，总天数为记录中的日期数
        # 计算每种故障每天的发生次数
        daily_faults = data.groupby('日期')[y_variables].sum()
        total_days = data['日期'].nunique()
        # 对每一天计算是否有任何故障发生
        any_faults_per_day = daily_faults.any(axis=1)

        # 计算有故障发生的天数
        unique_fault_days = any_faults_per_day.sum()

        # 计算修正后的设备综合故障率
        corrected_composite_fault_rate = (unique_fault_days / total_days) * 100 if total_days else 0

        # 计算物料推送效率
        material_push_efficiency = (data['物料推送数'].sum() / (
                    data['物料推送数'].sum() + data['物料待抓取数'].sum())) * 100

        # 计算填装效率
        filling_efficiency = (data['填装检测数'].sum() / data['物料推送数'].sum()) * 100 if data[
            '物料推送数'].sum() else 0

        # 计算加盖效率
        capping_efficiency = (data['加盖数'].sum() / data['填装检测数'].sum()) * 100 if data['填装检测数'].sum() else 0

        # 计算拧盖效率
        screwing_efficiency = (data['拧盖数'].sum() / data['加盖数'].sum()) * 100 if data['加盖数'].sum() else 0

        # 计算设备综合故障率
        # 修正设备综合故障率的计算

        # 提取所有设备故障的列名
        y_variables = data.columns[data.columns.str.contains('故障')]

        # 计算生产总周期数
        total_production_cycles = data['填装检测数'].sum()

        # 计算合格率
        acceptance_rate = (total_production / (total_production + total_defects) * 100) if (
                    total_production + total_defects) else 0

        additional_data = {
            "日期": k,
            "总运行时间": total_operation_time,
            "总生产数量": total_production,
            "总不合格数量": total_defects,
            "平均生产效率": average_efficiency,
            "设备综合故障率": corrected_composite_fault_rate,
            "物料推送效率": material_push_efficiency,
            "填装效率": filling_efficiency,
            "加盖效率": capping_efficiency,
            "拧盖效率": screwing_efficiency,
            "生产总周期数": total_production_cycles,
            "合格率": acceptance_rate
        }

        df = pd.concat([df, pd.DataFrame(additional_data, index=[count])])
        count = count + 1
    df['操作人员编号'] = tt['操作人员编号'].iloc[N]
    df['工龄'] = tt['工龄'].iloc[N]
    df['生产线编号'] = tt['生产线编号'].iloc[N]
    N = N + 1
    alldf = pd.concat([alldf, df])

# %%
alldf

# %%
alldf.reset_index(inplace=True, drop=True)
alldf

# %%
alldf.to_csv('Q3_alldf.csv', index=None)

# %%
alldf = pd.read_csv('Q3_alldf.csv')

# %%
tt1 = alldf.groupby('工龄')[['合格率', '总生产数量']].mean()

# %%
tt1['总生产数量'] = tt1['总生产数量'] / alldf['日期'].nunique()

# %%
tt1.reset_index(inplace=True)

# %%
tt1

# %%
tt1.to_csv('Q3_2.csv', index=None)

# %%
X = alldf[['总运行时间', '总生产数量', '总不合格数量', '平均生产效率',
           '设备综合故障率', '物料推送效率', '填装效率', '加盖效率', '拧盖效率', '生产总周期数', '合格率']]

Y = alldf['工龄']

# %%
X

# %%
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# %%


# 假设已经定义了 X, Y

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, Y)
LR_y_pred = model.predict(X)

# 输出线性回归模型评价结果
print('线性回归评价结果：')
# 计算均方误差和R^2评价
print("MSE", mean_squared_error(Y, LR_y_pred))
print("R^2", r2_score(Y, LR_y_pred))

axs[0][0].scatter(Y, LR_y_pred, color='blue')
axs[0][0].plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=2)
axs[0][0].set_xlabel('实际值')
axs[0][0].set_ylabel('预测值')
axs[0][0].set_title('线性回归')

# 构建决策树回归模型
from sklearn.tree import DecisionTreeRegressor

tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X, Y)
tree_y_pred = tree_model.predict(X)

# 输出决策树模型评价结果
print('决策树模型评价结果：')
# 计算均方误差和R^2评价
print("MSE", mean_squared_error(Y, tree_y_pred))
print("R^2", r2_score(Y, tree_y_pred))

axs[0][1].scatter(Y, tree_y_pred, color='green')
axs[0][1].plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=2)
axs[0][1].set_xlabel('实际值')
axs[0][1].set_ylabel('预测值')
axs[0][1].set_title('决策树')

# 构建随机森林回归模型
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, Y)
rf_y_pred = rf_model.predict(X)

# 输出随机森林模型评价结果
print('随机森林模型评价结果：')
# 计算均方误差和R^2评价
print("MSE", mean_squared_error(Y, rf_y_pred))
print("R^2", r2_score(Y, rf_y_pred))

axs[0][2].scatter(Y, rf_y_pred, color='red')
axs[0][2].plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=2)
axs[0][2].set_xlabel('实际值')
axs[0][2].set_ylabel('预测值')
axs[0][2].set_title('随机森林')

# 构建支持向量机回归模型
from sklearn.svm import SVR

svm_model = SVR()
svm_model.fit(X, Y)
svm_y_pred = svm_model.predict(X)

# 输出支持向量机模型评价结果
print('支持向量机模型评价结果：')
# 计算均方误差和R^2评价
print("MSE", mean_squared_error(Y, svm_y_pred))
print("R^2", r2_score(Y, svm_y_pred))

axs[1][0].scatter(Y, svm_y_pred, color='purple')
axs[1][0].plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=2)
axs[1][0].set_xlabel('实际值')
axs[1][0].set_ylabel('预测值')
axs[1][0].set_title('支持向量机')

# 构建神经网络回归模型
from sklearn.neural_network import MLPRegressor

mlp_model = MLPRegressor(random_state=42)
mlp_model.fit(X, Y)
mlp_y_pred = mlp_model.predict(X)

# 输出神经网络模型评价结果
print('神经网络模型评价结果：')
# 计算均方误差和R^2评价
print("MSE", mean_squared_error(Y, mlp_y_pred))
print("R^2", r2_score(Y, mlp_y_pred))

axs[1][1].scatter(Y, mlp_y_pred, color='orange')
axs[1][1].plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=2)
axs[1][1].set_xlabel('实际值')
axs[1][1].set_ylabel('预测值')
axs[1][1].set_title('神经网络')

plt.tight_layout()
plt.show()

# %% [markdown]
# 算法：
# 线性回归（最小二乘法）
#
# 变量：
# 自变量X：{ 总运行时间，总生产数量，总不合格数量，平均生产效率，设备综合故障率，物料推送效率，填装效率，加盖效率，拧盖效率，生产总周期数，合格率 }；因变量Y：{ 工龄 }
#
# https://www.spsspro.com/s/84af164888584fd4917d33afa9b6b2e4

# %%
import sys

# %%
import shap
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

# 初始化SHAP模型
explainer = shap.Explainer(tree_model)

# 计算SHAP值
shap_values = explainer(X)

# %%
shap_df = pd.DataFrame(shap_values.values, columns=X.columns)

# %%
# 输出每个特征的SHAP值（绝对值）
features = []
abs_mean_shap_values = []
for i, feature in enumerate(X.columns):
    abs_mean_shap_value = np.abs(shap_df.values[:, i]).mean()
    features.append(feature)
    abs_mean_shap_values.append(abs_mean_shap_value)
aa = pd.DataFrame(features, columns=['features'])
aa['shap_values'] = abs_mean_shap_values
aa.sort_values(by='shap_values', ascending=False)

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
# Matplotlib中设置字体-黑体，解决Matplotlib中文乱码问题
plt.rcParams['axes.unicode_minus'] = False
# 解决Matplotlib坐标轴负号'-'显示为方块的问题
sns.set(font='SimHei')
# Seaborn中设置字体-黑体，解决Seaborn中文乱码问题
a_sorted = aa.sort_values(by='shap_values', ascending=False)

# 设置图表大小，例如设置为宽10英寸，高6英寸
plt.figure(figsize=(8, 8))
# 使用Seaborn绘制柱状图
sns.barplot(x='shap_values', y='features', data=a_sorted.reset_index(), orient='h')

# 显示图形
plt.show()

# %%

# 初始化SHAP模型
explainer = shap.Explainer(tree_model)

shap.initjs()  # 初始化JS
shap_values = explainer.shap_values(X)  # 计算每个样本的每个特征的SHAP值

# %%
print(1)
shap.summary_plot(shap_values, X[X.columns])

# %%


# %%



