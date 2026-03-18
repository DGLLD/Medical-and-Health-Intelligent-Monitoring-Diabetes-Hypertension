"""
简化版训练脚本 - 生成模拟数据并训练模型
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import joblib
import os

# 创建目录
os.makedirs('models', exist_ok=True)
os.makedirs('scalers', exist_ok=True)
os.makedirs('imputers', exist_ok=True)

print("="*60)
print("健康风险预测系统 - 模型训练")
print("="*60)

# 1. 生成模拟的糖尿病数据
print("\n📊 生成模拟糖尿病数据集...")
np.random.seed(42)
n_samples = 1000

# 创建特征
age = np.random.normal(50, 15, n_samples).clip(18, 90)
bmi = np.random.normal(28, 5, n_samples).clip(15, 50)
glucose = np.random.normal(120, 30, n_samples).clip(60, 300)
blood_pressure = np.random.normal(130, 20, n_samples).clip(80, 200)
insulin = np.random.normal(80, 40, n_samples).clip(0, 300)
pregnancies = np.random.poisson(2, n_samples)
dpf = np.random.exponential(0.5, n_samples)

# 创建标签（基于规则生成）
risk_score = (age/100)*0.2 + (bmi/40)*0.3 + (glucose/200)*0.5
diabetes = (risk_score + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)

# 创建DataFrame
df_diabetes = pd.DataFrame({
    'Pregnancies': pregnancies,
    'Glucose': glucose,
    'BloodPressure': blood_pressure,
    'SkinThickness': np.random.normal(25, 8, n_samples).clip(5, 60),
    'Insulin': insulin,
    'BMI': bmi,
    'DiabetesPedigreeFunction': dpf,
    'Age': age,
    'Outcome': diabetes
})

df_diabetes.to_csv('diabetes_dataset.csv', index=False)
print(f"✅ 已生成模拟糖尿病数据集: {df_diabetes.shape}")

# 2. 生成模拟的高血压数据
print("\n📊 生成模拟高血压数据集...")
n_samples_ht = 1000

age_ht = np.random.normal(55, 15, n_samples_ht).clip(18, 90)
gender = np.random.randint(0, 2, n_samples_ht)
height = np.random.normal(170, 10, n_samples_ht).clip(140, 200)
weight = np.random.normal(75, 15, n_samples_ht).clip(40, 150)
ap_hi = np.random.normal(130, 20, n_samples_ht).clip(90, 200)
ap_lo = np.random.normal(85, 15, n_samples_ht).clip(60, 120)
cholesterol = np.random.randint(1, 4, n_samples_ht)
gluc = np.random.randint(1, 4, n_samples_ht)
smoke = np.random.binomial(1, 0.2, n_samples_ht)
alco = np.random.binomial(1, 0.1, n_samples_ht)
active = np.random.binomial(1, 0.7, n_samples_ht)

# 生成标签（基于血压值）
cardio = ((ap_hi > 140) | (ap_lo > 90)).astype(int)

df_hypertension = pd.DataFrame({
    'id': range(n_samples_ht),
    'age': age_ht * 365,  # 转换为天数
    'gender': gender,
    'height': height,
    'weight': weight,
    'ap_hi': ap_hi,
    'ap_lo': ap_lo,
    'cholesterol': cholesterol,
    'gluc': gluc,
    'smoke': smoke,
    'alco': alco,
    'active': active,
    'cardio': cardio
})

df_hypertension.to_csv('hypertension_dataset.csv', sep=';', index=False)
print(f"✅ 已生成模拟高血压数据集: {df_hypertension.shape}")

# 3. 训练糖尿病模型
print("\n" + "="*50)
print("🧠 训练糖尿病预测模型")
print("="*50)

# 准备数据
X_dm = df_diabetes.drop('Outcome', axis=1)
y_dm = df_diabetes['Outcome']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_dm, y_dm, test_size=0.2, random_state=42)

# 创建并训练 imputer（关键修改：先fit）
imputer_dm = SimpleImputer(strategy='median')
imputer_dm.fit(X_train)  # 👈 使用训练数据fit imputer

# 使用 imputer 处理数据
X_train_imputed = imputer_dm.transform(X_train)
X_test_imputed = imputer_dm.transform(X_test)

# 标准化
scaler_dm = StandardScaler()
X_train_scaled = scaler_dm.fit_transform(X_train_imputed)
X_test_scaled = scaler_dm.transform(X_test_imputed)

# 训练
model_dm = LogisticRegression(max_iter=1000, random_state=42)
model_dm.fit(X_train_scaled, y_train)

# 评估
train_acc = model_dm.score(X_train_scaled, y_train)
test_acc = model_dm.score(X_test_scaled, y_test)
print(f"训练集准确率: {train_acc:.2%}")
print(f"测试集准确率: {test_acc:.2%}")

# 保存模型（保存已经fit的imputer）
diabetes_model_data = {
    'model': model_dm,
    'scaler': scaler_dm,
    'imputer': imputer_dm,  # 👈 保存已经fit的imputer
    'feature_names': X_dm.columns.tolist(),
    'model_version': '1.0'
}
joblib.dump(diabetes_model_data, 'models/diabetes_model.pkl')
joblib.dump(scaler_dm, 'scalers/diabetes_scaler.pkl')
joblib.dump(imputer_dm, 'imputers/diabetes_imputer.pkl')  # 👈 保存已经fit的imputer
print("✅ 糖尿病模型已保存到 models/ 文件夹")

# 4. 训练高血压模型
print("\n" + "="*50)
print("🧠 训练高血压预测模型")
print("="*50)

# 特征工程
df_ht = df_hypertension.copy()
df_ht['bmi'] = df_ht['weight'] / ((df_ht['height']/100) ** 2)
df_ht['age_years'] = df_ht['age'] / 365

feature_cols = ['age_years', 'gender', 'bmi', 'ap_hi', 'ap_lo', 
                'cholesterol', 'gluc', 'smoke', 'alco', 'active']
X_ht = df_ht[feature_cols]
y_ht = df_ht['cardio']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_ht, y_ht, test_size=0.2, random_state=42)

# 高血压模型不需要imputer（数据完整），直接标准化
scaler_ht = StandardScaler()
X_train_scaled = scaler_ht.fit_transform(X_train)
X_test_scaled = scaler_ht.transform(X_test)

# 训练
model_ht = LogisticRegression(max_iter=1000, random_state=42)
model_ht.fit(X_train_scaled, y_train)

# 评估
train_acc = model_ht.score(X_train_scaled, y_train)
test_acc = model_ht.score(X_test_scaled, y_test)
print(f"训练集准确率: {train_acc:.2%}")
print(f"测试集准确率: {test_acc:.2%}")

# 保存模型
hypertension_model_data = {
    'model': model_ht,
    'scaler': scaler_ht,
    'feature_names': feature_cols,
    'model_version': '1.0'
}
joblib.dump(hypertension_model_data, 'models/hypertension_model.pkl')
joblib.dump(scaler_ht, 'scalers/hypertension_scaler.pkl')
print("✅ 高血压模型已保存到 models/ 文件夹")

print("\n" + "="*60)
print("🎉 训练完成！")
print("="*60)
print("\n生成的文件：")
print("  📁 models/")
print("    📄 diabetes_model.pkl")
print("    📄 hypertension_model.pkl")
print("  📁 scalers/")
print("    📄 diabetes_scaler.pkl")
print("    📄 hypertension_scaler.pkl")
print("  📁 imputers/")
print("    📄 diabetes_imputer.pkl")
print("\n数据集文件：")
print("  📄 diabetes_dataset.csv")
print("  📄 hypertension_dataset.csv")
print("\n现在可以运行测试脚本了：")
print("  python test_model.py")