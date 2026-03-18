#!/usr/bin/env python3
"""
📋 文件名称: app.py
🎯 功能描述: Flask Web应用入口 - 整合队友的登录系统 + 您的预测功能
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_cors import CORS
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入您的预测模块
from src.predict import predict_all_with_available_data

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production-2024'
CORS(app)

# ==================== 数据库配置导入 ====================
try:
    from db_config import DatabaseConfig
    db = DatabaseConfig(
        host=os.getenv('DB_HOST', '127.0.0.1'),
        port=int(os.getenv('DB_PORT', 3306)),
        user=os.getenv('DB_USER', 'root'),
        password=os.getenv('DB_PASSWORD', '111111'),#若是mysql密码不相同 需要修改mysql密码
        database=os.getenv('DB_NAME', 'test_db')
    )
except ImportError as e:
    print(f"⚠️ 警告: {e}")
    db = None

# ==================== 页面路由定义 ====================

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
    """
    服务页面 - 需要登录才能访问
    """
    # 检查用户是否已登录
    if 'user_id' not in session:
        print("666666")
        return redirect(url_for('login_page'))
    return render_template('service.html')

# ==================== 用户认证API ====================

@app.route('/api/login', methods=['POST'])
def api_login():
    """用户登录接口"""
    try:
        if not db:
            raise Exception("数据库未初始化")
        
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': '请求数据为空'})
        
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        # 参数校验
        if len(username) < 3 or len(username) > 20:
            return jsonify({'success': False, 'message': '账号长度必须在3-20之间'})
        if len(password) < 3 or len(password) > 12:
            return jsonify({'success': False, 'message': '密码长度必须在3-12之间'})
        
        # 验证用户
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
        return jsonify({
            'success': False,
            'message': f'服务器错误：{str(e)}'
        }), 500

@app.route('/api/register', methods=['POST'])
def api_register():
    """用户注册接口"""
    import re
    
    try:
        if not db:
            raise Exception("数据库未初始化")
        
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': '请求数据为空'})
        
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        # 格式验证
        if not re.match(r'^[a-zA-Z0-9_]{3,20}$', username):
            return jsonify({
                'success': False, 
                'message': '账号格式不正确（3-20位字母/数字/下划线）'
            })
        if not re.match(r'^\d{3,12}$', password):
            return jsonify({
                'success': False, 
                'message': '密码格式不正确（3-12位纯数字）'
            })
        
        # 检查账号是否存在
        if db.user_exists(username):
            return jsonify({
                'success': False, 
                'message': '该账号已被注册'
            })
        
        # 添加用户
        hashed_pw = password  # 测试阶段临时方案
        success, message, user_id = db.add_user(username, hashed_pw)
        
        if success:
            return jsonify({
                'success': True,
                'message': message,
                'user_id': user_id
            })
        else:
            return jsonify({
                'success': False,
                'message': message
            })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'服务器错误：{str(e)}'
        }), 500

@app.route('/api/logout', methods=['POST'])
def api_logout():
    """退出登录"""
    session.clear()
    return jsonify({'success': True, 'message': '已退出登录'})

# ==================== 您的预测API ====================

@app.route('/api/predict', methods=['POST'])
def predict():
    """健康预测接口"""
    try:
        # 检查登录状态（可选，如果需要登录才能使用预测功能）
        if 'user_id' not in session:
            return jsonify({
                'success': False,
                'error': '请先登录'
            }), 401
        
        data = request.json
        health_data = data.get('health_data', {})
        user_id = session.get('user_id')
        
        # 调用模型预测
        result = predict_all_with_available_data(health_data)
        
        # 如果需要保存预测记录到数据库，可以在这里添加
        
        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ==================== 预留扩展接口 ====================

@app.route('/api/user/info', methods=['GET'])
def get_user_info():
    """获取当前用户信息"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '未登录'}), 401
    return jsonify({
        'success': True,
        'user': {
            'id': session['user_id'],
            'username': session.get('username')
        }
    })

# ==================== 启动入口 ====================

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 医疗风险评估平台 - Flask服务器启动...")
    print("=" * 60)
    
    # 确保数据库表存在
    if db:
        try:
            print("\n📦 检查数据库配置...")
            db.create_database_if_not_exists()
            db.create_user_table()
            print("✅ 数据库初始化完成")
        except Exception as e:
            print(f"⚠️ 数据库初始化失败：{e}")
    
    # 启动Flask应用
    print("\n🌐 访问地址:")
    print("   - 首页：http://localhost:5000/")
    print("   - 登录页：http://localhost:5000/login")
    print("   - 注册页：http://localhost:5000/register")
    print("   - 服务页：http://localhost:5000/service")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )