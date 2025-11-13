import pandas as pd

# ========================
# 1. 读取两个文件
# ========================
file_metrics = "NTD Annual Data Metrics 2022-2024.csv"
file_main = "NTD Annual Data 2022-2024.csv"   # 暂时可能用不上

df = pd.read_csv(file_metrics, low_memory=False)

# ========================
# 2. 选择分析年份（你可以改）
# ========================
YEAR = 2023
df = df[df["Report Year"] == YEAR]

# ========================
# 3. 挑选计算需要的列
# ========================
required_cols = ["Agency", "State", "Mode", "Mode Name", "Mode VOMS"]
df = df[required_cols].copy()

# 去掉没有 VOMS 的行
df = df[df["Mode VOMS"].notnull()]

# ========================
# 4. 定义每种交通模式的典型容量（可调整）
# ========================
capacity_map = {
    "MB": 45,     # Bus
    "RB": 60,     # Rapid Bus/BRT
    "TB": 40,     # Trolleybus
    "LR": 150,    # Light Rail
    "SR": 100,    # Streetcar
    "HR": 1000,   # Heavy Rail / Subway
    "CR": 1500,   # Commuter Rail
    "YR": 800,    # Hybrid Rail (可调整)
    "BR": 1000,   # Bus Rapid Transit (如果数据出现)
}

# 未识别的模式默认给一个低值（你也可以 raise error）
df["Capacity"] = df["Mode"].map(capacity_map).fillna(40)

# ========================
# 5. 计算单行记录的峰值承载量
# ========================
df["PeakCapacity"] = df["Mode VOMS"] * df["Capacity"]

# ========================
# 6. 按州汇总
# ========================
state_summary = df.groupby("State").agg(
    Total_VOMS=("Mode VOMS", "sum"),
    Peak_Capacity=("PeakCapacity", "sum")
).reset_index()

# ========================
# 7. 按承载量排序
# ========================
state_summary = state_summary.sort_values("Peak_Capacity", ascending=False)

# ========================
# 8. 保存结果到 CSV 文件
# ========================
output_file = f"peak_capacity_by_state_{YEAR}.csv"
state_summary.to_csv(output_file, index=False)
print(f"完整结果已保存到: {output_file}")
print("\n前20个州的结果:")
print(state_summary.head(20))
