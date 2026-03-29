import dashscope
import os
import json

# 直接配置API密钥（仅用于测试）
dashscope.api_key = ""

def generate_health_report(user_data, prediction_result):
    """
    根据用户数据和预测结果生成个性化健康报告
    """
    # 检查API密钥是否配置
    if not dashscope.api_key:
        return "⚠️ 暂未配置AI健康顾问服务。建议咨询专业医生获取个性化建议。"
    
    # 构建用户数据描述
    user_info = []
    if user_data.get('age'):
        user_info.append(f"年龄：{user_data['age']}岁")
    if user_data.get('ap_hi'):
        user_info.append(f"收缩压：{user_data['ap_hi']} mmHg")
    if user_data.get('ap_lo'):
        user_info.append(f"舒张压：{user_data['ap_lo']} mmHg")
    if user_data.get('glucose'):
        user_info.append(f"血糖：{user_data['glucose']} mg/dL")
    if user_data.get('bmi'):
        user_info.append(f"BMI：{user_data['bmi']}")
    if user_data.get('hba1c'):
        user_info.append(f"糖化血红蛋白：{user_data['hba1c']}%")
    if user_data.get('pregnancies') is not None and user_data['pregnancies'] > 0:
        user_info.append(f"怀孕次数：{user_data['pregnancies']}次")
    
    # 获取预测结果
    ht_risk = prediction_result.get('hypertension', {}).get('risk', '未评估')
    ht_conf = prediction_result.get('hypertension', {}).get('confidence', 0) * 100
    db_risk = prediction_result.get('diabetes', {}).get('risk', '未评估')
    db_conf = prediction_result.get('diabetes', {}).get('confidence', 0) * 100
    
    # 构建提示词
    prompt = f"""你是一位温暖、专业的健康顾问。请根据以下用户健康数据，生成一份简洁、友好的个性化健康建议报告。

【用户健康数据】
{chr(10).join(user_info)}

【风险评估结果】
- 高血压风险：{ht_risk}（置信度{ht_conf:.0f}%）
- 糖尿病风险：{db_risk}（置信度{db_conf:.0f}%）

请生成一份包含以下内容的健康建议报告：
1. 🩺 风险解读：用通俗易懂的语言解释评估结果
2. 🥗 生活建议：针对性的饮食、运动、作息建议
3. 🏥 就医提醒：说明什么情况下需要就医
4. 💚 鼓励话语：温和、正向的鼓励

要求：
- 语言亲切友好，像朋友聊天一样
- 字数控制在200字以内
- 分点清晰，便于阅读
- 不要使用专业术语，用大白话
- 如果是高风险，语气要温和不恐吓；如果是低风险，也要提醒保持良好习惯"""

    try:
        response = dashscope.Generation.call(
            model='qwen-turbo',
            messages=[{'role': 'user', 'content': prompt}],
            result_format='message',
            temperature=0.7,
            max_tokens=800
        )
        
        report = response.output.choices[0].message.content
        return report
        
    except Exception as e:
        print(f"大模型调用失败：{e}")
        return "📋 健康建议：保持良好作息，均衡饮食，定期监测血压血糖。如有不适请及时就医。"