import pulp as pl
import pandas as pd

# 工人总数、生产线数和每日班次数
# 工人总数 这个很好计算 10个生产线，每天3班，每周上7天，需要10*7*3个班次，一个工人7天休息2天，每天上一斑，那么意味着需要10*7*3/5=42个工人
num_workers = 42
num_days = 7  # 假设是一周的排班
num_shifts = 3  # 早、中、晚班
shifts = ["Morning", "Afternoon", "Night"]

# 工龄、合格率和每日生产数量
experience_levels = {
    1: {"rate": 99.94394852, "quantity": 21555902.62, "proportion": 0.2},
    2: {"rate": 99.9477221, "quantity": 21145479.83, "proportion": 0.2},
    3: {"rate": 99.90503896, "quantity": 20774300.23, "proportion": 0.1},
    4: {"rate": 99.90342866, "quantity": 21189861.83, "proportion": 0.2},
    5: {"rate": 99.92651386, "quantity": 21198921.44, "proportion": 0.2},
    6: {"rate": 99.96382347, "quantity": 21589128.21, "proportion": 0.1}
}
# 初始化问题
model = pl.LpProblem("Worker_Scheduling", pl.LpMaximize)

# 决策变量: x_{ij} 表示工人 i 在第 j 天的班次
x = pl.LpVariable.dicts("shift",
                        ((i, j, k) for i in range(1, num_workers + 1)
                         for j in range(1, num_days + 1)
                         for k in shifts),
                        cat='Binary')  # Binary variable: 1 if worker i is in shift k on day j, else 0

# 目标函数：尽量最大化合格率乘以生产数量
objective = pl.lpSum(x[i, j, k] * experience_levels[exp]["rate"] * experience_levels[exp]["quantity"]
                     for i in range(1, num_workers + 1)
                     for j in range(1, num_days + 1)
                     for k in shifts
                     for exp in experience_levels
                     if i % len(experience_levels) + 1 == exp)

model += objective, "Total Quality Production"

# 每个工人每周工作5天，休息2天
for i in range(1, num_workers + 1):
    model += pl.lpSum(x[i, j, k] for j in range(1, num_days + 1) for k in shifts) == 5

# 每个班次每天需要10个不同的人
for j in range(1, num_days + 1):
    for k in shifts:
        model += pl.lpSum(x[i, j, k] for i in range(1, num_workers + 1)) == 10

# 工龄分布比例约束：例如，工龄1的工人占总工人数的20%
for exp, details in experience_levels.items():
    workers_in_exp = int(details["proportion"] * num_workers)
    model += pl.lpSum(x[i, j, k] for i in range(1, num_workers + 1)
                      for j in range(1, num_days + 1)
                      for k in shifts if i % len(experience_levels) + 1 == exp) == workers_in_exp * 5
# 求解模型
model.solve()


# 创建表1数据
table1_data = {f"B{i:03d}": [] for i in range(1, num_workers + 1)}
dates = list(range(1, num_days + 1))
ddd = {
    'M': '早',
    'A': '午',
    'N': '晚'
}
# 将决策变量值转换为班次或休息（早、中、晚、休）
for j in dates:
    for i in range(1, num_workers + 1):
        shifts_assigned = [k for k in shifts if pl.value(x[i, j, k]) == 1]
        if shifts_assigned:
            table1_data[f"B{i:03d}"].append(ddd[shifts_assigned[0][0]])  # 只取班次首字母
        else:
            table1_data[f"B{i:03d}"].append("休")

# 创建DataFrame
table1 = pd.DataFrame(table1_data, index=dates)

table1 = pd.concat([table1] * 52, ignore_index=True)


table1['日期'] = range(1, 52 * 7 + 1)

import random

# 创建表2数据
table2_columns = ["日期", "班次"] + [f"M3{i:02d}" for i in range(1, 11)]
table2_data = []

# 跟踪每天每个工人的班次分配
worker_shifts_per_day = {j: [] for j in dates}

# 按照班次和日期整理数据
for j in dates:
    for k in shifts:
        row = [j, k]
        workers_in_shift = [f"B{i:03d}" for i in range(1, num_workers + 1) if pl.value(x[i, j, k]) == 1]
        workers_assigned = len(workers_in_shift)

        # 记录当天已分配班次的工人
        worker_shifts_per_day[j].extend(workers_in_shift)

        # 检查是否有足够的工人，如果不够则从未分配的工人中随机选择
        if workers_assigned < 10:
            unassigned_workers = [f"B{i:03d}" for i in range(1, num_workers + 1) if
                                  f"B{i:03d}" not in worker_shifts_per_day[j]]
            workers_to_add = random.sample(unassigned_workers, 10 - workers_assigned)
            workers_in_shift.extend(workers_to_add)

        # 确保列表只有10个工人
        row.extend(workers_in_shift[:10])
        table2_data.append(row)

# 创建DataFrame
table2 = pd.DataFrame(table2_data, columns=table2_columns)
table2['班次'] = table2['班次'].map({
    'Morning': '早',
    'Afternoon': '午',
    'Night': '晚'
})


table2 = pd.concat([table2] * 52, ignore_index=True)

ee = []
for i in range(1, 52 * 7 + 1):
    ee.append(i)
    ee.append(i)
    ee.append(i)

# %%
table2['日期'] = ee


table1.to_csv("result4-1.xlsx", index=None)
table2.to_csv("result4-2.xlsx", index=None)





