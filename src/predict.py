import joblib
import numpy as np
import pandas as pd
import os
import re  # 🔴【方案三】导入正则表达式模块

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)

# 加载所有模型
model_h = joblib.load(os.path.join(project_dir, 'models', 'hypertension_model.pkl'))
scaler_h = joblib.load(os.path.join(project_dir, 'models', 'hypertension_scaler.pkl'))

# 专业模型（含病史特征）- 原糖尿病模型1
model_professional = joblib.load(os.path.join(project_dir, 'models', 'diabetes_model_general.pkl'))
scaler_professional = joblib.load(os.path.join(project_dir, 'models', 'diabetes_scaler_general.pkl'))

# 通用模型（基础特征）- 原糖尿病模型2
model_general = joblib.load(os.path.join(project_dir, 'models', 'diabetes_model_pima.pkl'))
scaler_general = joblib.load(os.path.join(project_dir, 'models', 'diabetes_scaler_pima.pkl'))

# 加载平均值
avg_values = pd.read_csv(os.path.join(project_dir, 'models', 'avg_values.csv'), index_col=0).squeeze().to_dict()

# smoking_history 映射
smoking_map = {'never': 0, 'No Info': 1, 'current': 2, 'former': 3, 'not current': 4}

# 🔴【方案三】数据清洗函数
def clean_input_value(value, field_type='int'):
    """清洗输入值，防止特殊字符注入"""
    if value is None or value == '':
        return None
    
    try:
        # 如果是字符串，先清洗
        if isinstance(value, str):
            # 只保留数字、小数点、负号
            value = re.sub(r'[^0-9.-]', '', value)
            # 去除多余的小数点
            if value.count('.') > 1:
                parts = value.split('.')
                value = parts[0] + '.' + ''.join(parts[1:])
        
        # 转换为浮点数
        num_value = float(value)
        
        # 根据字段类型处理
        if field_type == 'int':
            # 转为整数
            return int(num_value)
        elif field_type == 'float':
            # 保留一位小数
            return round(num_value, 1)
        else:
            return num_value
            
    except (ValueError, TypeError):
        return None

def predict_hypertension(ap_hi, ap_lo, gluc, cholesterol, age, weight, height):
    """高血压预测（需要7个特征）"""
    features = np.array([[ap_hi, ap_lo, gluc, cholesterol, age, weight, height]])
    feature_names = ['ap_hi', 'ap_lo', 'gluc', 'cholesterol', 'age', 'weight', 'height']
    features_df = pd.DataFrame(features, columns=feature_names)
    features_scaled = scaler_h.transform(features_df)
    pred = model_h.predict(features_scaled)[0]
    prob = model_h.predict_proba(features_scaled)[0]
    confidence = min(float(max(prob)), 0.95)  # 置信度上限95%
    return {
        'risk': '高风险' if pred == 1 else '低风险',
        'confidence': confidence,
        'prediction': int(pred)
    }

def predict_diabetes_professional(age, bmi, blood_glucose, hba1c, hypertension, smoking):
    """
    专业糖尿病模型预测
    训练特征: age, bmi, blood_glucose_level, HbA1c_level, hypertension, smoking_history
    """
    # 处理smoking（如果是文本就转换）
    if isinstance(smoking, str):
        smoking = smoking_map.get(smoking, 1)
    
    features = np.array([[age, bmi, blood_glucose, hba1c, hypertension, float(smoking)]])
    feature_names = ['age', 'bmi', 'blood_glucose_level', 'HbA1c_level', 'hypertension', 'smoking_history']
    features_df = pd.DataFrame(features, columns=feature_names)
    features_scaled = scaler_professional.transform(features_df)
    pred = model_professional.predict(features_scaled)[0]
    prob = model_professional.predict_proba(features_scaled)[0]
    confidence = min(float(max(prob)), 0.95)  # 置信度上限95%
    return {
        'risk': '高风险' if pred == 1 else '低风险',
        'confidence': confidence,
        'prediction': int(pred),
        'model': '专业模型'
    }

def predict_diabetes_general(glucose, bloodpressure, bmi, age, pregnancies, insulin):
    """
    通用糖尿病模型预测
    训练特征: Glucose, BloodPressure, BMI, Age, Pregnancies, Insulin
    """
    features = np.array([[glucose, bloodpressure, bmi, age, pregnancies, insulin]])
    feature_names = ['Glucose', 'BloodPressure', 'BMI', 'Age', 'Pregnancies', 'Insulin']
    features_df = pd.DataFrame(features, columns=feature_names)
    features_scaled = scaler_general.transform(features_df)
    pred = model_general.predict(features_scaled)[0]
    prob = model_general.predict_proba(features_scaled)[0]
    confidence = min(float(max(prob)), 0.95)  # 置信度上限95%
    return {
        'risk': '高风险' if pred == 1 else '低风险',
        'confidence': confidence,
        'prediction': int(pred),
        'model': '通用模型'
    }

def predict_all_with_available_data(user_data):
    """
    根据用户提供的数据，智能选择模型进行预测
    保证永远有结果：通用模型作为保底方案
    """
    # 🔴【方案三】数据清洗
    cleaned_data = {}
    
    # 定义各字段的类型
    int_fields = ['age', 'ap_hi', 'ap_lo', 'glucose', 'pregnancies', 
                  'weight', 'height', 'cholesterol', 'smoking', 'hypertension',
                  'gluc']  # 整数型字段
    float_fields = ['bmi', 'hba1c', 'insulin']  # 浮点型字段
    
    for key, value in user_data.items():
        if key in float_fields:
            cleaned = clean_input_value(value, 'float')
        elif key in int_fields:
            cleaned = clean_input_value(value, 'int')
        else:
            cleaned = clean_input_value(value)
        
        if cleaned is not None:
            cleaned_data[key] = cleaned
    
    # 如果清洗后没有数据，使用默认值
    if not cleaned_data:
        cleaned_data = {'age': 45, 'ap_hi': 120, 'glucose': 100}
    
    # 用清洗后的数据替换原数据
    user_data = cleaned_data
    # 🔴【方案三结束】

    result = {
        'hypertension': None,
        'diabetes': None,
        'explanation': {}
    }
    
    # === 高血压预测（只要有收缩压就能预测）===
    if 'ap_hi' in user_data:
        h_features = [
            user_data.get('ap_hi', avg_values.get('ap_hi', 120)),
            user_data.get('ap_lo', avg_values.get('ap_lo', 80)),
            user_data.get('gluc', avg_values.get('gluc', 1)),
            user_data.get('cholesterol', avg_values.get('cholesterol', 1)),
            user_data.get('age_days', avg_values.get('age_h', 20000)),
            user_data.get('weight', avg_values.get('weight', 70)),
            user_data.get('height', avg_values.get('height', 165))
        ]
        result['hypertension'] = predict_hypertension(*h_features)
        result['explanation']['hypertension'] = '基于您提供的数据估算'
    else:
        result['explanation']['hypertension'] = '请填写收缩压以进行高血压预测'
    
    # === 糖尿病预测（智能选择模型）===
    
    # 获取血糖值（用户可能有多种字段名）
    glucose_value = None
    if 'glucose' in user_data:
        glucose_value = user_data['glucose']
    elif 'blood_glucose_level' in user_data:
        glucose_value = user_data['blood_glucose_level']
    
    if not glucose_value:
        result['explanation']['diabetes'] = '请填写血糖值以进行糖尿病预测'
        return result
    
    # 判断是否使用专业模型（需要hba1c + 至少2项其他特征）
    has_hba1c = 'hba1c' in user_data
    other_features_count = 0
    if 'hypertension' in user_data:
        other_features_count += 1
    if 'smoking' in user_data:
        other_features_count += 1
    if 'bmi' in user_data and user_data['bmi']:  # BMI有值才算
        other_features_count += 1
    
    # 专业模型触发条件：有hba1c + 至少2项其他特征
    use_professional = has_hba1c and other_features_count >= 2
    
    if use_professional:
        # 使用专业模型 - 只传入专业模型需要的6个特征
        pro_features = [
            user_data.get('age', avg_values.get('age_d1', 45)),  # age
            user_data.get('bmi', avg_values.get('bmi_d1', 25)),  # bmi
            glucose_value,                                        # blood_glucose_level
            user_data.get('hba1c', avg_values.get('hba1c', 5.5)), # HbA1c_level
            user_data.get('hypertension', 0),                    # hypertension
            user_data.get('smoking', 1)                           # smoking_history
        ]
        result['diabetes'] = predict_diabetes_professional(*pro_features)
        result['explanation']['diabetes'] = '使用专业模型评估（基于血糖、糖化血红蛋白和病史）'
    else:
        # 使用通用模型（保底方案）- 只传入通用模型需要的6个特征
        # 从用户数据中提取通用模型需要的特征，缺失的用平均值填充
        gen_features = [
            glucose_value,  # Glucose - 血糖值
            user_data.get('bloodpressure', 
                         user_data.get('ap_lo', 
                                      avg_values.get('bloodpressure', 70))),  # BloodPressure
            user_data.get('bmi', avg_values.get('bmi_d2', 30)),  # BMI
            user_data.get('age', avg_values.get('age_d2', 40)),  # Age
            user_data.get('pregnancies', avg_values.get('pregnancies', 3)),  # Pregnancies
            user_data.get('insulin', avg_values.get('insulin', 80))  # Insulin
        ]
        result['diabetes'] = predict_diabetes_general(*gen_features)
        result['explanation']['diabetes'] = '使用通用模型评估（基于基础指标）'
    
    return result

# 以下是交互式测试代码（可选）
if __name__ == "__main__":
    print("\n=== 糖尿病预测模型测试 ===")
    print("1. 测试专业模型场景")
    print("2. 测试通用模型场景")
    print("3. 测试混合数据场景")
    
    choice = input("\n请选择测试场景 (1/2/3): ").strip()
    
    if choice == '1':
        # 专业模型测试数据
        test_data = {
            'age': 45,
            'bmi': 26.5,
            'blood_glucose_level': 140,
            'hba1c': 6.2,
            'hypertension': 1,
            'smoking': 2,
            'ap_hi': 130,
            'ap_lo': 85
        }
        print("\n测试数据:", test_data)
        result = predict_all_with_available_data(test_data)
        print("\n预测结果:")
        print(f"高血压: {result['hypertension']['risk']} (置信度: {result['hypertension']['confidence']:.3f})")
        print(f"糖尿病: {result['diabetes']['risk']} (置信度: {result['diabetes']['confidence']:.3f})")
        print(f"使用模型: {result['diabetes']['model']}")
        print(f"说明: {result['explanation']['diabetes']}")
        
    elif choice == '2':
        # 通用模型测试数据
        test_data = {
            'age': 45,
            'glucose': 140,
            'bloodpressure': 85,
            'bmi': 26.5,
            'pregnancies': 2,
            'insulin': 100,
            'ap_hi': 130,
            'ap_lo': 85
        }
        print("\n测试数据:", test_data)
        result = predict_all_with_available_data(test_data)
        print("\n预测结果:")
        print(f"高血压: {result['hypertension']['risk']} (置信度: {result['hypertension']['confidence']:.3f})")
        print(f"糖尿病: {result['diabetes']['risk']} (置信度: {result['diabetes']['confidence']:.3f})")
        print(f"使用模型: {result['diabetes']['model']}")
        print(f"说明: {result['explanation']['diabetes']}")
        
    elif choice == '3':
        # 混合数据（同时满足两个模型的部分条件）
        test_data = {
            'age': 45,
            'bmi': 26.5,
            'glucose': 140,
            'blood_glucose_level': 140,
            'hba1c': 6.2,
            'hypertension': 1,
            'ap_hi': 130,
            'ap_lo': 85
        }
        print("\n测试数据:", test_data)
        result = predict_all_with_available_data(test_data)
        print("\n预测结果:")
        print(f"高血压: {result['hypertension']['risk']} (置信度: {result['hypertension']['confidence']:.3f})")
        print(f"糖尿病: {result['diabetes']['risk']} (置信度: {result['diabetes']['confidence']:.3f})")
        print(f"使用模型: {result['diabetes']['model']}")
        print(f"说明: {result['explanation']['diabetes']}")