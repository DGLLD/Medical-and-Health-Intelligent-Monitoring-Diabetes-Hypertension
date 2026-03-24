#!/usr/bin/env python3
"""
📋 文件名称: app.py
🎯 功能描述: Flask Web应用入口
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_cors import CORS
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.predict import predict_all_with_available_data
from db_config import DatabaseConfig

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production-2024')
CORS(app)

# 初始化数据库配置（从环境变量读取）
db = DatabaseConfig()

# ==================== 页面路由 ====================

@app.route('/')
def index():
    """首页"""
    return render_template('index.html')

@app.route('/login')
def login_page():
    """登录页面"""
    return render_template('login.html')

@app.route('/register')
def register_page():
    """注册页面"""
    return render_template('register.html')

@app.route('/service')
def service_page():
    """服务页面"""
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return render_template('service.html')

# ==================== API接口 ====================

@app.route('/api/login', methods=['POST'])
def api_login():
    """用户登录接口"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if len(username) < 3 or len(username) > 20:
            return jsonify({'success': False, 'message': '账号长度必须在3-20之间'})
        if len(password) < 3 or len(password) > 12:
            return jsonify({'success': False, 'message': '密码长度必须在3-12之间'})
        
        result = db.verify_user(username, password)
        
        if result['success']:
            session['user_id'] = result['user_id']
            session['username'] = username
            return jsonify({
                'success': True,
                'user_id': result['user_id'],
                'message': result['message']
            })
        else:
            return jsonify({
                'success': False,
                'message': result['message']
            })
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'服务器错误：{str(e)}'}), 500

@app.route('/api/register', methods=['POST'])
def api_register():
    """用户注册接口"""
    import re
    
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if not re.match(r'^[a-zA-Z0-9_]{3,20}$', username):
            return jsonify({'success': False, 'message': '账号格式不正确（3-20位字母/数字/下划线）'})
        if not re.match(r'^\d{3,12}$', password):
            return jsonify({'success': False, 'message': '密码格式不正确（3-12位纯数字）'})
        
        if db.user_exists(username):
            return jsonify({'success': False, 'message': '该账号已被注册'})
        
        success, message, user_id = db.add_user(username, password)
        
        if success:
            return jsonify({'success': True, 'message': message, 'user_id': user_id})
        else:
            return jsonify({'success': False, 'message': message})
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'服务器错误：{str(e)}'}), 500

@app.route('/api/logout', methods=['POST'])
def api_logout():
    """退出登录"""
    session.clear()
    return jsonify({'success': True, 'message': '已退出登录'})

@app.route('/api/predict', methods=['POST'])
def predict():
    """健康预测接口"""
    try:
        if 'user_id' not in session:
            return jsonify({'success': False, 'error': '请先登录'}), 401
        
        data = request.json
        health_data = data.get('health_data', {})
        
        result = predict_all_with_available_data(health_data)
        
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/user/info', methods=['GET'])
def get_user_info():
    """获取用户信息"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '未登录'}), 401
    return jsonify({
        'success': True,
        'user': {'id': session['user_id'], 'username': session.get('username')}
    })

# ==================== 启动入口 ====================

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 医疗风险评估平台 - Flask服务器启动...")
    print("=" * 60)
    
    try:
        db.create_database_if_not_exists()
        db.create_user_table()
        print("✅ 数据库初始化完成")
    except Exception as e:
        print(f"⚠️ 数据库初始化失败：{e}")
    
    port = int(os.getenv('APP_PORT', 5000))
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False
    )