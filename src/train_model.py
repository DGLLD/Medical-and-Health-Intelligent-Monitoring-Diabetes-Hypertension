import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

os.makedirs('models', exist_ok=True)

def print_dataset_info(name, df, y_col):
    """打印数据集基本信息"""
    print(f"\n=== {name} 数据集信息 ===")
    print(f"原始数据量: {len(df)}")
    print(f"特征列: {list(df.columns)}")
    print(f"标签分布:\n{df[y_col].value_counts(normalize=True).round(3)}")

def clean_hypertension(df):
    """清洗高血压数据集"""
    original_len = len(df)
    
    # 删除重复行
    df = df.drop_duplicates()
    
    # 异常值过滤（医学合理范围）
    df = df[
        (df['ap_hi'] >= 50) & (df['ap_hi'] <= 250) &  # 收缩压合理范围
        (df['ap_lo'] >= 30) & (df['ap_lo'] <= 150) &  # 舒张压合理范围
        (df['ap_hi'] > df['ap_lo']) &  # 收缩压必须大于舒张压
        (df['weight'] >= 20) & (df['weight'] <= 200) &  # 体重合理范围
        (df['height'] >= 100) & (df['height'] <= 250) &  # 身高合理范围
        (df['age'] >= 0) & (df['age'] <= 36500)  # 年龄0-100岁（天）
    ]
    
    cleaned_len = len(df)
    print(f"高血压数据清洗: {original_len} → {cleaned_len} (删除 {original_len-cleaned_len} 行)")
    return df

def clean_diabetes1(df):
    """清洗糖尿病数据集1"""
    original_len = len(df)
    
    # 删除重复行
    df = df.drop_duplicates()
    
    # 异常值过滤
    df = df[
        (df['age'] >= 0) & (df['age'] <= 120) &  # 年龄合理范围
        (df['bmi'] >= 10) & (df['bmi'] <= 50) &  # BMI合理范围
        (df['blood_glucose_level'] >= 50) & (df['blood_glucose_level'] <= 500) &  # 血糖合理范围
        (df['HbA1c_level'] >= 4) & (df['HbA1c_level'] <= 15)  # 糖化血红蛋白合理范围
    ]
    
    cleaned_len = len(df)
    print(f"糖尿病数据集1清洗: {original_len} → {cleaned_len} (删除 {original_len-cleaned_len} 行)")
    return df

def clean_diabetes2(df):
    """清洗糖尿病数据集2"""
    original_len = len(df)
    
    # 删除重复行
    df = df.drop_duplicates()
    
    # 异常值过滤
    df = df[
        (df['Glucose'] >= 50) & (df['Glucose'] <= 500) &  # 血糖合理范围
        (df['BloodPressure'] >= 40) & (df['BloodPressure'] <= 200) &  # 血压合理范围
        (df['BMI'] >= 10) & (df['BMI'] <= 50) &  # BMI合理范围
        (df['Age'] >= 0) & (df['Age'] <= 120) &  # 年龄合理范围
        (df['Pregnancies'] >= 0) & (df['Pregnancies'] <= 20) &  # 怀孕次数合理范围
        (df['Insulin'] >= 0) & (df['Insulin'] <= 500)  # 胰岛素合理范围
    ]
    
    cleaned_len = len(df)
    print(f"糖尿病数据集2清洗: {original_len} → {cleaned_len} (删除 {original_len-cleaned_len} 行)")
    return df

# ==================== 高血压模型训练 ====================
print("="*60)
print("开始训练高血压模型...")
print("="*60)

# 读取数据
df_hypertension = pd.read_csv('data/高血压数据集.csv', sep=';')
print_dataset_info("高血压原始数据", df_hypertension, 'cardio')

# 数据清洗
df_hypertension = clean_hypertension(df_hypertension)

# 准备特征和标签
feature_cols_h = ['ap_hi', 'ap_lo', 'gluc', 'cholesterol', 'age', 'weight', 'height']
X_h = df_hypertension[feature_cols_h]
y_h = df_hypertension['cardio']

# 缺失值填充
X_h = X_h.fillna(X_h.mean())

# 特征归一化
scaler_h = StandardScaler()
X_h_scaled = scaler_h.fit_transform(X_h)

# 划分训练集和测试集
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_h_scaled, y_h, test_size=0.2, random_state=42, stratify=y_h
)

# 训练模型
model_h = LogisticRegression(max_iter=1000, class_weight='balanced')
model_h.fit(X_train_h, y_train_h)

# 评估模型
train_score = model_h.score(X_train_h, y_train_h)
test_score = model_h.score(X_test_h, y_test_h)
print(f"\n高血压模型 - 训练集准确率: {train_score:.3f}")
print(f"高血压模型 - 测试集准确率: {test_score:.3f}")

# 保存模型
joblib.dump(model_h, 'models/hypertension_model.pkl')
joblib.dump(scaler_h, 'models/hypertension_scaler.pkl')
print("✅ 高血压模型已保存")

# ==================== 糖尿病模型1训练 ====================
print("\n" + "="*60)
print("开始训练糖尿病模型1（专业模型）...")
print("="*60)

# 读取数据
df_diabetes1 = pd.read_csv('data/糖尿病数据集1.csv')
print_dataset_info("糖尿病数据集1原始数据", df_diabetes1, 'diabetes')

# 数据清洗
df_diabetes1 = clean_diabetes1(df_diabetes1)

# 准备特征和标签
feature_cols_d1 = ['age', 'bmi', 'blood_glucose_level', 'HbA1c_level', 'hypertension', 'smoking_history']
X_d1 = df_diabetes1[feature_cols_d1].copy()
y_d1 = df_diabetes1['diabetes']

# 处理 smoking_history 编码（文本转数字）
smoking_map = {'never': 0, 'No Info': 1, 'current': 2, 'former': 3, 'not current': 4}
X_d1['smoking_history'] = X_d1['smoking_history'].map(smoking_map).fillna(1)

# 缺失值填充
X_d1 = X_d1.fillna(X_d1.mean())

# 特征归一化
scaler_d1 = StandardScaler()
X_d1_scaled = scaler_d1.fit_transform(X_d1)

# 划分训练集和测试集
X_train_d1, X_test_d1, y_train_d1, y_test_d1 = train_test_split(
    X_d1_scaled, y_d1, test_size=0.2, random_state=42, stratify=y_d1
)

# 训练模型
model_d1 = LogisticRegression(max_iter=1000, class_weight='balanced')
model_d1.fit(X_train_d1, y_train_d1)

# 评估模型
train_score = model_d1.score(X_train_d1, y_train_d1)
test_score = model_d1.score(X_test_d1, y_test_d1)
print(f"\n糖尿病模型1 - 训练集准确率: {train_score:.3f}")
print(f"糖尿病模型1 - 测试集准确率: {test_score:.3f}")

# 保存模型
joblib.dump(model_d1, 'models/diabetes_model_professional.pkl')
joblib.dump(scaler_d1, 'models/diabetes_scaler_professional.pkl')
print("✅ 糖尿病专业模型已保存")

# ==================== 糖尿病模型2训练 ====================
print("\n" + "="*60)
print("开始训练糖尿病模型2（通用模型）...")
print("="*60)

# 读取数据
df_diabetes2 = pd.read_csv('data/糖尿病数据集2.csv')
print_dataset_info("糖尿病数据集2原始数据", df_diabetes2, 'Outcome')

# 数据清洗
df_diabetes2 = clean_diabetes2(df_diabetes2)

# 准备特征和标签
feature_cols_d2 = ['Glucose', 'BloodPressure', 'BMI', 'Age', 'Pregnancies', 'Insulin']
X_d2 = df_diabetes2[feature_cols_d2]
y_d2 = df_diabetes2['Outcome']

# 缺失值填充
X_d2 = X_d2.fillna(X_d2.mean())

# 特征归一化
scaler_d2 = StandardScaler()
X_d2_scaled = scaler_d2.fit_transform(X_d2)

# 划分训练集和测试集
X_train_d2, X_test_d2, y_train_d2, y_test_d2 = train_test_split(
    X_d2_scaled, y_d2, test_size=0.2, random_state=42, stratify=y_d2
)

# 训练模型
model_d2 = LogisticRegression(max_iter=1000, class_weight='balanced')
model_d2.fit(X_train_d2, y_train_d2)

# 评估模型
train_score = model_d2.score(X_train_d2, y_train_d2)
test_score = model_d2.score(X_test_d2, y_test_d2)
print(f"\n糖尿病模型2 - 训练集准确率: {train_score:.3f}")
print(f"糖尿病模型2 - 测试集准确率: {test_score:.3f}")

# 保存模型
joblib.dump(model_d2, 'models/diabetes_model_general.pkl')
joblib.dump(scaler_d2, 'models/diabetes_scaler_general.pkl')
print("✅ 糖尿病通用模型已保存")

# ==================== 计算人群平均值 ====================
print("\n" + "="*60)
print("计算人群平均值...")
print("="*60)

avg_values = {
    # 高血压相关
    'ap_hi': float(df_hypertension['ap_hi'].mean()),
    'ap_lo': float(df_hypertension['ap_lo'].mean()),
    'gluc': float(df_hypertension['gluc'].mean()),
    'cholesterol': float(df_hypertension['cholesterol'].mean()),
    'age_h': float(df_hypertension['age'].mean()),
    'weight': float(df_hypertension['weight'].mean()),
    'height': float(df_hypertension['height'].mean()),
    
    # 糖尿病模型1相关
    'age_d1': float(df_diabetes1['age'].mean()),
    'bmi_d1': float(df_diabetes1['bmi'].mean()),
    'blood_glucose': float(df_diabetes1['blood_glucose_level'].mean()),
    'hba1c': float(df_diabetes1['HbA1c_level'].mean()),
    
    # 糖尿病模型2相关
    'age_d2': float(df_diabetes2['Age'].mean()),
    'bmi_d2': float(df_diabetes2['BMI'].mean()),
    'glucose': float(df_diabetes2['Glucose'].mean()),
    'bloodpressure': float(df_diabetes2['BloodPressure'].mean()),
    'pregnancies': float(df_diabetes2['Pregnancies'].mean()),
    'insulin': float(df_diabetes2['Insulin'].mean())
}

# 保存平均值
pd.Series(avg_values).to_csv('models/avg_values.csv')
print("\n✅ 人群平均值已保存至 models/avg_values.csv")
print("\n各特征平均值:")
for key, value in avg_values.items():
    print(f"  {key}: {value:.2f}")

print("\n" + "="*60)
print("✅ 所有模型训练完成！")
print("="*60)
print("\n生成的文件:")
print("  - models/hypertension_model.pkl")
print("  - models/hypertension_scaler.pkl")
print("  - models/diabetes_model_professional.pkl")
print("  - models/diabetes_scaler_professional.pkl")
print("  - models/diabetes_model_general.pkl")
print("  - models/diabetes_scaler_general.pkl")
print("  - models/avg_values.csv")