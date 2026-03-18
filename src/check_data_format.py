import pandas as pd

print("=== 检查糖尿病数据集1 ===")
# 尝试不同的分隔符
try:
    df1_comma = pd.read_csv('data/糖尿病数据集1.csv')
    print("用逗号分隔读取成功！")
    print("列名:", list(df1_comma.columns))
    print("前2行:")
    print(df1_comma.head(2))
except:
    print("逗号分隔失败")

try:
    df1_tab = pd.read_csv('data/糖尿病数据集1.csv', sep='\t')
    print("\n用制表符分隔读取成功！")
    print("列名:", list(df1_tab.columns))
    print("前2行:")
    print(df1_tab.head(2))
except:
    print("制表符分隔失败")

print("\n=== 检查糖尿病数据集2 ===")
try:
    df2 = pd.read_csv('data/糖尿病数据集2.csv')
    print("读取成功！")
    print("列名:", list(df2.columns))
    print("前2行:")
    print(df2.head(2))
except Exception as e:
    print(f"读取失败: {e}")

print("\n=== 检查高血压数据集 ===")
try:
    df_h = pd.read_csv('data/高血压数据集.csv', sep=';')
    print("读取成功！")
    print("列名:", list(df_h.columns))
    print("前2行:")
    print(df_h.head(2))
except Exception as e:
    print(f"读取失败: {e}")