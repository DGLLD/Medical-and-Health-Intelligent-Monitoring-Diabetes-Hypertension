import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import predict_all_with_available_data

print("开始批量测试智能预测系统...\n")

# 测试用例
test_cases = [
    {
        'name': '案例1：完整体检报告（高风险）',
        'data': {
            'ap_hi': 160, 'ap_lo': 100, 'gluc': 2, 'cholesterol': 3,
            'age_days': 25000, 'weight': 85, 'height': 165,
            'glucose': 180, 'bloodpressure': 95, 'bmi': 32,
            'age': 68, 'pregnancies': 3, 'insulin': 200,
            'hba1c': 8.5, 'hypertension': 1, 'smoking': 2
        }
    },
    {
        'name': '案例2：部分数据（只有基础指标）',
        'data': {
            'age': 45,
            'ap_hi': 130,
            'glucose': 140
        }
    },
    {
        'name': '案例3：适合Pima模型的数据',
        'data': {
            'glucose': 148, 'bloodpressure': 72, 'bmi': 33.6,
            'age': 50, 'pregnancies': 6, 'insulin': 0
        }
    },
    {
        'name': '案例4：适合通用模型的数据',
        'data': {
            'age': 54, 'bmi': 27.32, 'blood_glucose': 80,
            'hba1c': 6.6, 'hypertension': 0, 'smoking': 1
        }
    },
    {
        'name': '案例5：快速评估（只有3项）',
        'data': {
            'age': 30,
            'ap_hi': 115,
            'glucose': 90
        }
    }
]

for i, case in enumerate(test_cases, 1):
    print(f"\n{'='*60}")
    print(f"测试 {i}: {case['name']}")
    print(f"{'='*60}")
    print("输入数据:", case['data'])
    
    result = predict_all_with_available_data(case['data'])
    
    print("\n预测结果:")
    if result['hypertension']:
        print(f"高血压: {result['hypertension']['risk']} "
              f"(置信度: {result['hypertension']['confidence']:.3f})")
    else:
        print(f"高血压: {result['explanation']['hypertension']}")
    
    if result['diabetes']:
        print(f"糖尿病: {result['diabetes']['risk']} "
              f"(置信度: {result['diabetes']['confidence']:.3f}) "
              f"[{result['diabetes'].get('model', '')}]")
    else:
        print(f"糖尿病: {result['explanation']['diabetes']}")
    
    print(f"说明: {result['explanation']}")