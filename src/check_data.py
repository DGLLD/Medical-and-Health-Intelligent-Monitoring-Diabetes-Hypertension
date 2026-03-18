import pandas as pd

print("=== 高血压数据集 ===")
df_h = pd.read_csv('data/高血压数据集.csv')
print("列名:", list(df_h.columns))
print("前2行数据:")
print(df_h.head(2))
print()

print("=== 糖尿病数据集1 ===")
df_d1 = pd.read_csv('data/糖尿病数据集1.csv')
print("列名:", list(df_d1.columns))
print("前2行数据:")
print(df_d1.head(2))
print()

print("=== 糖尿病数据集2 ===")
df_d2 = pd.read_csv('data/糖尿病数据集2.csv')
print("列名:", list(df_d2.columns))
print("前2行数据:")
print(df_d2.head(2))