import joblib
import numpy as np
import pandas as pd
import os
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)

# 加载模型
model_h = joblib.load(os.path.join(project_dir, 'models', 'hypertension_model.pkl'))
scaler_h = joblib.load(os.path.join(project_dir, 'models', 'hypertension_scaler.pkl'))

model_professional = joblib.load(os.path.join(project_dir, 'models', 'diabetes_model_general.pkl'))
scaler_professional = joblib.load(os.path.join(project_dir, 'models', 'diabetes_scaler_general.pkl'))

model_general = joblib.load(os.path.join(project_dir, 'models', 'diabetes_model_pima.pkl'))
scaler_general = joblib.load(os.path.join(project_dir, 'models', 'diabetes_scaler_pima.pkl'))

# 加载平均值
avg_values = pd.read_csv(os.path.join(project_dir, 'models', 'avg_values.csv'), index_col=0).squeeze().to_dict()

# 检测模型特征数量
def get_model_feature_count(model, scaler):
    """获取模型期望的特征数量"""
    try:
        return scaler.n_features_in_
    except:
        return None

h_feature_count = get_model_feature_count(model_h, scaler_h)
d1_feature_count = get_model_feature_count(model_professional, scaler_professional)
d2_feature_count = get_model_feature_count(model_general, scaler_general)

# smoking_history 映射
smoking_map = {'never': 0, 'No Info': 1, 'current': 2, 'former': 3, 'not current': 4}

def safe_float_conversion(value, default):
    """安全地将输入值转换为浮点数"""
    if value is None or value == '':
        return default
    try:
        if isinstance(value, str):
            value = value.strip()
            if value == '':
                return default
        return float(value)
    except (ValueError, TypeError):
        return default

def compute_bmi(weight, height):
    """计算BMI"""
    if weight and height and height > 0:
        return weight / ((height / 100) ** 2)
    return None

def create_hypertension_features(user_data):
    """创建高血压模型的特征"""
    ap_hi = safe_float_conversion(user_data.get('ap_hi'), avg_values.get('ap_hi', 120))
    ap_lo = safe_float_conversion(user_data.get('ap_lo'), avg_values.get('ap_lo', 80))
    gluc = safe_float_conversion(user_data.get('gluc'), avg_values.get('gluc', 1))
    cholesterol = safe_float_conversion(user_data.get('cholesterol'), avg_values.get('cholesterol', 1))
    age = safe_float_conversion(user_data.get('age_days'), avg_values.get('age_h', 20000))
    weight = safe_float_conversion(user_data.get('weight'), avg_values.get('weight', 70))
    height = safe_float_conversion(user_data.get('height'), avg_values.get('height', 165))
    
    # 基础7个特征
    base_features = [ap_hi, ap_lo, gluc, cholesterol, age, weight, height]
    
    # 如果模型期望14个特征，则添加交叉特征
    if h_feature_count == 14:
        bmi = compute_bmi(weight, height)
        if bmi is None:
            bmi = avg_values.get('bmi', 24)
        
        enhanced_features = [
            ap_hi, ap_lo, gluc, cholesterol, age, weight, height,
            age * ap_hi,
            bmi,
            bmi * ap_hi,
            gluc * cholesterol,
            ap_hi ** 2,
            ap_lo ** 2,
            age ** 2
        ]
        return enhanced_features
    
    return base_features

def create_diabetes_professional_features(user_data):
    """创建糖尿病专业模型的特征"""
    age = safe_float_conversion(user_data.get('age'), avg_values.get('age_d1', 45))
    bmi = safe_float_conversion(user_data.get('bmi'), avg_values.get('bmi_d1', 25))
    blood_glucose = safe_float_conversion(
        user_data.get('blood_glucose_level') or user_data.get('glucose'), 
        avg_values.get('blood_glucose', 100)
    )
    hba1c = safe_float_conversion(user_data.get('hba1c'), avg_values.get('hba1c', 5.5))
    hypertension = safe_float_conversion(user_data.get('hypertension'), 0)
    smoking = safe_float_conversion(user_data.get('smoking'), 1)
    
    if isinstance(smoking, str):
        smoking = smoking_map.get(smoking, 1)
    
    # 基础6个特征
    base_features = [age, bmi, blood_glucose, hba1c, hypertension, smoking]
    
    # 如果模型期望12个特征，则添加交叉特征
    if d1_feature_count == 12:
        enhanced_features = [
            age, bmi, blood_glucose, hba1c, hypertension, smoking,
            age * bmi,
            bmi * blood_glucose,
            blood_glucose * hba1c,
            hypertension * smoking,
            bmi ** 2,
            blood_glucose ** 2
        ]
        return enhanced_features
    
    return base_features

def create_diabetes_general_features(user_data):
    """创建糖尿病通用模型的特征"""
    glucose = safe_float_conversion(
        user_data.get('glucose') or user_data.get('blood_glucose_level'),
        avg_values.get('glucose', 120)
    )
    bloodpressure = safe_float_conversion(
        user_data.get('bloodpressure') or user_data.get('ap_lo'),
        avg_values.get('bloodpressure', 70)
    )
    bmi = safe_float_conversion(user_data.get('bmi'), avg_values.get('bmi_d2', 30))
    age = safe_float_conversion(user_data.get('age'), avg_values.get('age_d2', 40))
    pregnancies = safe_float_conversion(user_data.get('pregnancies'), avg_values.get('pregnancies', 3))
    insulin = safe_float_conversion(user_data.get('insulin'), avg_values.get('insulin', 80))
    
    # 基础6个特征
    base_features = [glucose, bloodpressure, bmi, age, pregnancies, insulin]
    
    # 如果模型期望13个特征，则添加交叉特征
    if d2_feature_count == 13:
        enhanced_features = [
            glucose, bloodpressure, bmi, age, pregnancies, insulin,
            glucose * bmi,
            glucose * age,
            bmi * age,
            pregnancies * glucose,
            glucose ** 2,
            bmi ** 2,
            age ** 2
        ]
        return enhanced_features
    
    return base_features

def predict_diabetes_professional(features):
    """专业糖尿病模型预测"""
    features_scaled = scaler_professional.transform([features])
    pred = model_professional.predict(features_scaled)[0]
    prob = model_professional.predict_proba(features_scaled)[0]
    raw_confidence = float(max(prob))
    
    # 置信度校准：医疗场景不应出现100%或超过95%
    if raw_confidence > 0.95:
        # 超过95%时，压缩到85-92%区间
        confidence = 0.85 + (raw_confidence - 0.95) * 0.7
        confidence = min(confidence, 0.92)
    else:
        confidence = raw_confidence
    
    return {
        'risk': '高风险' if pred == 1 else '低风险',
        'confidence': confidence,
        'prediction': int(pred),
        'model': '专业模型'
    }

def predict_diabetes_general(features):
    """通用糖尿病模型预测"""
    features_scaled = scaler_general.transform([features])
    pred = model_general.predict(features_scaled)[0]
    prob = model_general.predict_proba(features_scaled)[0]
    raw_confidence = float(max(prob))
    
    # 通用模型上限95%
    confidence = min(raw_confidence, 0.95)
    
    return {
        'risk': '高风险' if pred == 1 else '低风险',
        'confidence': confidence,
        'prediction': int(pred),
        'model': '通用模型'
    }

def predict_hypertension(features):
    """高血压预测"""
    features_scaled = scaler_h.transform([features])
    pred = model_h.predict(features_scaled)[0]
    prob = model_h.predict_proba(features_scaled)[0]
    raw_confidence = float(max(prob))
    
    # 高血压模型上限92%
    confidence = min(raw_confidence, 0.92)
    
    return {
        'risk': '高风险' if pred == 1 else '低风险',
        'confidence': confidence,
        'prediction': int(pred)
    }

def predict_all_with_available_data(user_data):
    """
    根据用户提供的数据，智能选择模型进行预测
    """
    result = {
        'hypertension': None,
        'diabetes': None,
        'explanation': {}
    }
    
    # === 高血压预测 ===
    if 'ap_hi' in user_data:
        h_features = create_hypertension_features(user_data)
        result['hypertension'] = predict_hypertension(h_features)
        result['explanation']['hypertension'] = '基于您提供的数据估算'
    else:
        result['explanation']['hypertension'] = '请填写收缩压以进行高血压预测'
    
    # === 糖尿病预测 ===
    glucose_value = user_data.get('glucose') or user_data.get('blood_glucose_level')
    
    if not glucose_value:
        result['explanation']['diabetes'] = '请填写血糖值以进行糖尿病预测'
        return result
    
    # 方案一：基于数据完整度自动触发
    # 专业模型需要的特征
    professional_features = {
        'hba1c': user_data.get('hba1c'),
        'age': user_data.get('age'),
        'bmi': user_data.get('bmi'),
        'hypertension': user_data.get('hypertension'),
        'smoking': user_data.get('smoking')
    }
    
    # 计算已填写的专业模型特征数量（排除None和0）
    filled_count = 0
    for key, value in professional_features.items():
        if key == 'hba1c':
            if value is not None and value != '' and float(value) > 0:
                filled_count += 1
        elif key == 'age':
            if value is not None and value != '':
                filled_count += 1
        elif key == 'bmi':
            if value is not None and value != '' and float(value) > 0:
                filled_count += 1
        elif key in ['hypertension', 'smoking']:
            if value is not None and value != '' and float(value) != 0:
                filled_count += 1
    
    # 触发专业模型的条件：
    # 1. 有糖化血红蛋白
    # 2. 已填写的专业模型特征数量 >= 3（或可调整）
    has_hba1c = professional_features['hba1c'] is not None and professional_features['hba1c'] != ''
    use_professional = has_hba1c and filled_count >= 3
    
    # 可选：如果用户没有填写糖化血红蛋白但有其他完整数据，仍使用通用模型
    if use_professional:
        d1_features = create_diabetes_professional_features(user_data)
        result['diabetes'] = predict_diabetes_professional(d1_features)
        result['explanation']['diabetes'] = '使用专业模型评估（基于血糖、糖化血红蛋白和病史）'
    else:
        d2_features = create_diabetes_general_features(user_data)
        result['diabetes'] = predict_diabetes_general(d2_features)
        if has_hba1c:
            result['explanation']['diabetes'] = '已检测到糖化血红蛋白，但数据不足，仍使用通用模型评估'
        else:
            result['explanation']['diabetes'] = '使用通用模型评估（基于基础指标）'
    
    return result