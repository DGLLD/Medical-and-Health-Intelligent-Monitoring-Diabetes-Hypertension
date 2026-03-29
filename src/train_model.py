import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

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
    df = df.drop_duplicates()
    df = df[
        (df['ap_hi'] >= 50) & (df['ap_hi'] <= 250) &
        (df['ap_lo'] >= 30) & (df['ap_lo'] <= 150) &
        (df['ap_hi'] > df['ap_lo']) &
        (df['weight'] >= 20) & (df['weight'] <= 200) &
        (df['height'] >= 100) & (df['height'] <= 250) &
        (df['age'] >= 0) & (df['age'] <= 438001)        
    ]
    cleaned_len = len(df)
    print(f"高血压数据清洗: {original_len} → {cleaned_len} (删除 {original_len-cleaned_len} 行)")
    return df

def clean_diabetes1(df):
    """清洗糖尿病数据集1"""
    original_len = len(df)
    df = df.drop_duplicates()
    df = df[
        (df['age'] >= 0) & (df['age'] <= 120) &
        (df['bmi'] >= 10) & (df['bmi'] <= 50) &
        (df['blood_glucose_level'] >= 50) & (df['blood_glucose_level'] <= 500) &
        (df['HbA1c_level'] >= 4) & (df['HbA1c_level'] <= 15)
    ]
    cleaned_len = len(df)
    print(f"糖尿病数据集1清洗: {original_len} → {cleaned_len} (删除 {original_len-cleaned_len} 行)")
    return df

def clean_diabetes2(df):
    """清洗糖尿病数据集2"""
    original_len = len(df)
    df = df.drop_duplicates()
    df = df[
        (df['Glucose'] >= 50) & (df['Glucose'] <= 500) &
        (df['BloodPressure'] >= 40) & (df['BloodPressure'] <= 200) &
        (df['BMI'] >= 10) & (df['BMI'] <= 50) &
        (df['Age'] >= 0) & (df['Age'] <= 120) &
        (df['Pregnancies'] >= 0) & (df['Pregnancies'] <= 20) &
        (df['Insulin'] >= 0) & (df['Insulin'] <= 500)
    ]
    cleaned_len = len(df)
    print(f"糖尿病数据集2清洗: {original_len} → {cleaned_len} (删除 {original_len-cleaned_len} 行)")
    return df

def create_interaction_features_hypertension(df):
    """为高血压数据创建特征交叉和多项式特征"""
    df = df.copy()
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    df['age_bp_interaction'] = df['age'] * df['ap_hi']
    df['bmi_bp_interaction'] = df['bmi'] * df['ap_hi']
    df['gluc_chol_interaction'] = df['gluc'] * df['cholesterol']
    df['ap_hi_squared'] = df['ap_hi'] ** 2
    df['ap_lo_squared'] = df['ap_lo'] ** 2
    df['age_squared'] = df['age'] ** 2
    return df

def create_interaction_features_diabetes1(df):
    """为糖尿病专业模型创建特征交叉"""
    df = df.copy()
    df['age_bmi_interaction'] = df['age'] * df['bmi']
    df['bmi_glucose_interaction'] = df['bmi'] * df['blood_glucose_level']
    df['glucose_hba1c_interaction'] = df['blood_glucose_level'] * df['HbA1c_level']
    df['hypertension_smoking_interaction'] = df['hypertension'] * df['smoking_history']
    df['bmi_squared'] = df['bmi'] ** 2
    df['glucose_squared'] = df['blood_glucose_level'] ** 2
    return df

def create_interaction_features_diabetes2(df):
    """为糖尿病通用模型创建特征交叉"""
    df = df.copy()
    df['glucose_bmi_interaction'] = df['Glucose'] * df['BMI']
    df['glucose_age_interaction'] = df['Glucose'] * df['Age']
    df['bmi_age_interaction'] = df['BMI'] * df['Age']
    df['pregnancies_glucose_interaction'] = df['Pregnancies'] * df['Glucose']
    df['glucose_squared'] = df['Glucose'] ** 2
    df['bmi_squared'] = df['BMI'] ** 2
    df['age_squared'] = df['Age'] ** 2
    return df

def train_with_grid_search(X_train, y_train, X_test, y_test, model_name):
    """使用网格搜索进行超参数调优"""
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['liblinear']
    }
    
    base_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    train_score = best_model.score(X_train, y_train)
    test_score = best_model.score(X_test, y_test)
    
    print(f"\n{model_name} 最优参数: {grid_search.best_params_}")
    print(f"{model_name} - 训练集准确率: {train_score:.4f}")
    print(f"{model_name} - 测试集准确率: {test_score:.4f}")
    
    return best_model

# ==================== 高血压模型训练 ====================
print("="*60)
print("开始训练高血压模型...")
print("="*60)

df_hypertension = pd.read_csv('data/高血压数据集.csv', sep=';')
print_dataset_info("高血压原始数据", df_hypertension, 'cardio')
df_hypertension = clean_hypertension(df_hypertension)

# 原始特征训练（保留原版）
feature_cols_h_base = ['ap_hi', 'ap_lo', 'gluc', 'cholesterol', 'age', 'weight', 'height']
X_h_base = df_hypertension[feature_cols_h_base]
y_h = df_hypertension['cardio']
X_h_base = X_h_base.fillna(X_h_base.mean())
scaler_h_base = StandardScaler()
X_h_base_scaled = scaler_h_base.fit_transform(X_h_base)
X_train_h_base, X_test_h_base, y_train_h_base, y_test_h_base = train_test_split(
    X_h_base_scaled, y_h, test_size=0.2, random_state=42, stratify=y_h
)
model_h_base = LogisticRegression(max_iter=1000, class_weight='balanced')
model_h_base.fit(X_train_h_base, y_train_h_base)
print(f"\n高血压模型（原始） - 测试集准确率: {model_h_base.score(X_test_h_base, y_test_h_base):.3f}")

# 优化特征训练
df_enhanced_h = create_interaction_features_hypertension(df_hypertension)
feature_cols_h_enhanced = [
    'ap_hi', 'ap_lo', 'gluc', 'cholesterol', 'age', 'weight', 'height',
    'age_bp_interaction', 'bmi', 'bmi_bp_interaction', 'gluc_chol_interaction',
    'ap_hi_squared', 'ap_lo_squared', 'age_squared'
]
X_h_enhanced = df_enhanced_h[feature_cols_h_enhanced]
X_h_enhanced = X_h_enhanced.fillna(X_h_enhanced.mean())
scaler_h_enhanced = StandardScaler()
X_h_enhanced_scaled = scaler_h_enhanced.fit_transform(X_h_enhanced)
X_train_h_enhanced, X_test_h_enhanced, y_train_h_enhanced, y_test_h_enhanced = train_test_split(
    X_h_enhanced_scaled, y_h, test_size=0.2, random_state=42, stratify=y_h
)

model_h_enhanced = train_with_grid_search(
    X_train_h_enhanced, y_train_h_enhanced, X_test_h_enhanced, y_test_h_enhanced, "高血压模型"
)

# 选择表现更好的模型
if model_h_enhanced.score(X_test_h_enhanced, y_test_h_enhanced) >= model_h_base.score(X_test_h_base, y_test_h_base):
    model_h = model_h_enhanced
    scaler_h = scaler_h_enhanced
    print("\n✅ 采用优化版高血压模型")
else:
    model_h = model_h_base
    scaler_h = scaler_h_base
    print("\n✅ 采用原版高血压模型")

joblib.dump(model_h, 'models/hypertension_model.pkl')
joblib.dump(scaler_h, 'models/hypertension_scaler.pkl')
print("✅ 高血压模型已保存")

# ==================== 糖尿病模型1训练 ====================
print("\n" + "="*60)
print("开始训练糖尿病模型1（专业模型）...")
print("="*60)

df_diabetes1 = pd.read_csv('data/糖尿病数据集1.csv')
print_dataset_info("糖尿病数据集1原始数据", df_diabetes1, 'diabetes')
df_diabetes1 = clean_diabetes1(df_diabetes1)

# 原始特征训练
feature_cols_d1_base = ['age', 'bmi', 'blood_glucose_level', 'HbA1c_level', 'hypertension', 'smoking_history']
X_d1_base = df_diabetes1[feature_cols_d1_base].copy()
y_d1 = df_diabetes1['diabetes']

smoking_map = {'never': 0, 'No Info': 1, 'current': 2, 'former': 3, 'not current': 4}
X_d1_base['smoking_history'] = X_d1_base['smoking_history'].map(smoking_map).fillna(1)
X_d1_base = X_d1_base.fillna(X_d1_base.mean())
scaler_d1_base = StandardScaler()
X_d1_base_scaled = scaler_d1_base.fit_transform(X_d1_base)
X_train_d1_base, X_test_d1_base, y_train_d1_base, y_test_d1_base = train_test_split(
    X_d1_base_scaled, y_d1, test_size=0.2, random_state=42, stratify=y_d1
)
model_d1_base = LogisticRegression(max_iter=1000, class_weight='balanced')
model_d1_base.fit(X_train_d1_base, y_train_d1_base)
print(f"\n糖尿病模型1（原始） - 测试集准确率: {model_d1_base.score(X_test_d1_base, y_test_d1_base):.3f}")

# 优化特征训练
df_enhanced_d1 = create_interaction_features_diabetes1(df_diabetes1)
feature_cols_d1_enhanced = [
    'age', 'bmi', 'blood_glucose_level', 'HbA1c_level', 'hypertension', 'smoking_history',
    'age_bmi_interaction', 'bmi_glucose_interaction', 'glucose_hba1c_interaction',
    'hypertension_smoking_interaction', 'bmi_squared', 'glucose_squared'
]
X_d1_enhanced = df_enhanced_d1[feature_cols_d1_enhanced]
X_d1_enhanced['smoking_history'] = X_d1_enhanced['smoking_history'].map(smoking_map).fillna(1)
X_d1_enhanced = X_d1_enhanced.fillna(X_d1_enhanced.mean())
scaler_d1_enhanced = StandardScaler()
X_d1_enhanced_scaled = scaler_d1_enhanced.fit_transform(X_d1_enhanced)
X_train_d1_enhanced, X_test_d1_enhanced, y_train_d1_enhanced, y_test_d1_enhanced = train_test_split(
    X_d1_enhanced_scaled, y_d1, test_size=0.2, random_state=42, stratify=y_d1
)

model_d1_enhanced = train_with_grid_search(
    X_train_d1_enhanced, y_train_d1_enhanced, X_test_d1_enhanced, y_test_d1_enhanced, "糖尿病专业模型"
)

# 选择表现更好的模型
if model_d1_enhanced.score(X_test_d1_enhanced, y_test_d1_enhanced) >= model_d1_base.score(X_test_d1_base, y_test_d1_base):
    model_d1 = model_d1_enhanced
    scaler_d1 = scaler_d1_enhanced
    print("\n✅ 采用优化版糖尿病专业模型")
else:
    model_d1 = model_d1_base
    scaler_d1 = scaler_d1_base
    print("\n✅ 采用原版糖尿病专业模型")

joblib.dump(model_d1, 'models/diabetes_model_general.pkl')
joblib.dump(scaler_d1, 'models/diabetes_scaler_general.pkl')
print("✅ 糖尿病专业模型已保存")

# ==================== 糖尿病模型2训练 ====================
print("\n" + "="*60)
print("开始训练糖尿病模型2（通用模型）...")
print("="*60)

df_diabetes2 = pd.read_csv('data/糖尿病数据集2.csv')
print_dataset_info("糖尿病数据集2原始数据", df_diabetes2, 'Outcome')
df_diabetes2 = clean_diabetes2(df_diabetes2)

# 原始特征训练
feature_cols_d2_base = ['Glucose', 'BloodPressure', 'BMI', 'Age', 'Pregnancies', 'Insulin']
X_d2_base = df_diabetes2[feature_cols_d2_base]
y_d2 = df_diabetes2['Outcome']
X_d2_base = X_d2_base.fillna(X_d2_base.mean())
scaler_d2_base = StandardScaler()
X_d2_base_scaled = scaler_d2_base.fit_transform(X_d2_base)
X_train_d2_base, X_test_d2_base, y_train_d2_base, y_test_d2_base = train_test_split(
    X_d2_base_scaled, y_d2, test_size=0.2, random_state=42, stratify=y_d2
)
model_d2_base = LogisticRegression(max_iter=1000, class_weight='balanced')
model_d2_base.fit(X_train_d2_base, y_train_d2_base)
print(f"\n糖尿病模型2（原始） - 测试集准确率: {model_d2_base.score(X_test_d2_base, y_test_d2_base):.3f}")

# 优化特征训练
df_enhanced_d2 = create_interaction_features_diabetes2(df_diabetes2)
feature_cols_d2_enhanced = [
    'Glucose', 'BloodPressure', 'BMI', 'Age', 'Pregnancies', 'Insulin',
    'glucose_bmi_interaction', 'glucose_age_interaction', 'bmi_age_interaction',
    'pregnancies_glucose_interaction', 'glucose_squared', 'bmi_squared', 'age_squared'
]
X_d2_enhanced = df_enhanced_d2[feature_cols_d2_enhanced]
X_d2_enhanced = X_d2_enhanced.fillna(X_d2_enhanced.mean())
scaler_d2_enhanced = StandardScaler()
X_d2_enhanced_scaled = scaler_d2_enhanced.fit_transform(X_d2_enhanced)
X_train_d2_enhanced, X_test_d2_enhanced, y_train_d2_enhanced, y_test_d2_enhanced = train_test_split(
    X_d2_enhanced_scaled, y_d2, test_size=0.2, random_state=42, stratify=y_d2
)

model_d2_enhanced = train_with_grid_search(
    X_train_d2_enhanced, y_train_d2_enhanced, X_test_d2_enhanced, y_test_d2_enhanced, "糖尿病通用模型"
)

# 选择表现更好的模型
if model_d2_enhanced.score(X_test_d2_enhanced, y_test_d2_enhanced) >= model_d2_base.score(X_test_d2_base, y_test_d2_base):
    model_d2 = model_d2_enhanced
    scaler_d2 = scaler_d2_enhanced
    print("\n✅ 采用优化版糖尿病通用模型")
else:
    model_d2 = model_d2_base
    scaler_d2 = scaler_d2_base
    print("\n✅ 采用原版糖尿病通用模型")

joblib.dump(model_d2, 'models/diabetes_model_pima.pkl')
joblib.dump(scaler_d2, 'models/diabetes_scaler_pima.pkl')
print("✅ 糖尿病通用模型已保存")

# ==================== 计算人群平均值 ====================
print("\n" + "="*60)
print("计算人群平均值...")
print("="*60)

avg_values = {
    'ap_hi': float(df_hypertension['ap_hi'].mean()),
    'ap_lo': float(df_hypertension['ap_lo'].mean()),
    'gluc': float(df_hypertension['gluc'].mean()),
    'cholesterol': float(df_hypertension['cholesterol'].mean()),
    'age_h': float(df_hypertension['age'].mean()),
    'weight': float(df_hypertension['weight'].mean()),
    'height': float(df_hypertension['height'].mean()),
    'bmi': float((df_hypertension['weight'] / ((df_hypertension['height']/100)**2)).mean()),
    
    'age_d1': float(df_diabetes1['age'].mean()),
    'bmi_d1': float(df_diabetes1['bmi'].mean()),
    'blood_glucose': float(df_diabetes1['blood_glucose_level'].mean()),
    'hba1c': float(df_diabetes1['HbA1c_level'].mean()),
    
    'age_d2': float(df_diabetes2['Age'].mean()),
    'bmi_d2': float(df_diabetes2['BMI'].mean()),
    'glucose': float(df_diabetes2['Glucose'].mean()),
    'bloodpressure': float(df_diabetes2['BloodPressure'].mean()),
    'pregnancies': float(df_diabetes2['Pregnancies'].mean()),
    'insulin': float(df_diabetes2['Insulin'].mean())
}

pd.Series(avg_values).to_csv('models/avg_values.csv')
print("\n✅ 人群平均值已保存至 models/avg_values.csv")
print("\n各特征平均值:")
for key, value in avg_values.items():
    print(f"  {key}: {value:.2f}")

print("\n" + "="*60)
print("✅ 所有模型训练完成！")
print("="*60)