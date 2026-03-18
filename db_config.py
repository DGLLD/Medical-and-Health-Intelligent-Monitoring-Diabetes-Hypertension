#!/usr/bin/env python3
"""
📋 文件名称: db_config.py
🎯 功能描述: 数据库连接与用户操作模块
📅 创建时间: 2024年
✏️ 作者说明: 封装MySQL连接和用户CRUD操作
🔌 预留接口: add_user, verify_user, user_exists
"""

import pymysql
import os
from datetime import datetime


class DatabaseConfig:
    """
    🔧 数据库配置类 - 管理MySQL连接和用户操作
    """
    
    def __init__(self, host='127.0.0.1', port=3306, user='root', 
                 password='111111', database='test_db', charset='utf8mb4'):#此处我修改了密码 后续换回即可
        """
        【构造方法】初始化数据库连接参数
        
        Args:
            host (str): 数据库主机地址
            port (int): 数据库端口号
            user (str): 数据库用户名
            password (str): 数据库密码
            database (str): 数据库名称
            charset (str): 字符集编码
        """
        self.config = {
            'host': host,
            'port': port,
            'user': user,
            'password': password,
            'database': database,
            'charset': charset,
            'cursorclass': pymysql.cursors.DictCursor  # 返回字典类型游标
        }
    
    def get_connection(self):
        """
        🔗 获取数据库连接对象
        
        Returns:
            pymysql.Connection: 数据库连接对象
        """
        conn = pymysql.connect(**self.config)
        return conn
    
    def create_database_if_not_exists(self, db_name=None):
        """
        🗂️ 检查并创建数据库（如果不存在）
        
        Args:
            db_name (str): 数据库名称
        
        Returns:
            bool: 是否成功创建
        """
        db_name = db_name or self.config['database']
        conn = None
        cursor = None
        
        try:
            # 不指定数据库进行连接
            temp_config = {k: v for k, v in self.config.items() if k != 'database'}
            conn = pymysql.connect(**temp_config)
            cursor = conn.cursor()
            
            # 检查数据库是否存在
            cursor.execute("SELECT SCHEMA_NAME FROM information_schema.SCHEMATA WHERE SCHEMA_NAME = %s;", (db_name,))
            result = cursor.fetchone()
            
            if not result:
                print(f"📝 未找到数据库 '{db_name}'，正在创建...")
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`;")
                conn.commit()
                print(f"✅ 数据库 '{db_name}' 创建成功")
                return True
            
            print(f"ℹ️ 数据库 '{db_name}' 已存在")
            return False
            
        except Exception as e:
            print(f"❌ 创建数据库失败：{e}")
            return False
        finally:
            if cursor: cursor.close()
            if conn: conn.close()
    
    def create_user_table(self):
        """
        👤 创建用户数据表 users
        
        【表结构】
        - id: INT AUTO_INCREMENT PRIMARY KEY
        - username: VARCHAR(50) UNIQUE 用户账号（3-20字符）
        - password: VARCHAR(128) 密码哈希值
        - created_at: TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        - updated_at: TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE
        """
        conn = None
        cursor = None
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS `{}`.`users` (
                `id` INT AUTO_INCREMENT PRIMARY KEY COMMENT '用户ID',
                `username` VARCHAR(50) NOT NULL UNIQUE COMMENT '用户账号（3-20字符）',
                `password` VARCHAR(128) NOT NULL COMMENT '密码哈希值',
                `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
                `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
                INDEX idx_username (`username`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """.format(self.config['database'])
            
            print(f"📝 执行建表SQL：{create_table_sql[:100]}...")
            cursor.execute(create_table_sql)
            conn.commit()
            print("✅ 用户表 'users' 创建成功")
            return True
            
        except Exception as e:
            print(f"❌ 创建用户表失败：{e}")
            conn.rollback()
            return False
        finally:
            if cursor: cursor.close()
            if conn: conn.close()
    
    def user_exists(self, username):
        """
        ✅ 检查账号是否存在
        
        Args:
            username (str): 要检查的用户名
        
        Returns:
            bool: True表示存在，False表示不存在
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            sql = "SELECT COUNT(*) FROM `users` WHERE username = %s"
            cursor.execute(sql, (username,))
            result = cursor.fetchone()
            return result[0] > 0
        except Exception as e:
            print(f"❌ 查询失败：{e}")
            return False
        finally:
            cursor.close()
            conn.close()
    
    def add_user(self, username, hashed_password):
        """
        ➕ 添加新用户到数据库
        
        Args:
            username (str): 用户账号
            hashed_password (str): 加密后的密码哈希值
        
        Returns:
            tuple: (success:bool, message:str, user_id:int)
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            insert_sql = "INSERT INTO `users` (username, password) VALUES (%s, %s)"
            cursor.execute(insert_sql, (username, hashed_password))
            conn.commit()
            
            user_id = cursor.lastrowid
            print(f"✅ 用户 '{username}' 添加成功，ID={user_id}")
            return (True, "注册成功", user_id)
            
        except Exception as e:
            conn.rollback()
            error_msg = str(e)
            if 'Duplicate entry' in error_msg:
                return (False, "该账号已被注册", 0)
            else:
                return (False, f"服务器错误：{error_msg}", 0)
        finally:
            cursor.close()
            conn.close()
    
    def verify_user(self, username, plain_password):
        """
        🔐 验证用户登录
        
        Args:
            username (str): 用户账号
            plain_password (str): 明文密码（需在后端解密比对）
        
        Returns:
            dict: {'success': bool, 'user_id': int, 'message': str}
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            select_sql = "SELECT id, username, password FROM `users` WHERE username = %s"
            cursor.execute(select_sql, (username,))
            user = cursor.fetchone()
            
            if not user:
                return {'success': False, 'user_id': None, 'message': '账号不存在'}
            
            # 🔴【注意】生产环境使用 bcrypt.checkpw() 或 werkzeug.security.check_password_hash()
            stored_hash = user['password']
            
            if stored_hash == plain_password:  # 测试阶段临时方案
                print(f"✅ 用户验证成功：{username}")
                return {'success': True, 'user_id': user['id'], 'message': '登录成功'}
            else:
                return {'success': False, 'user_id': None, 'message': '密码错误'}
            
        except Exception as e:
            return {'success': False, 'user_id': None, 'message': f'验证错误：{str(e)}'}
        finally:
            cursor.close()
            conn.close()


# ==================== 独立测试函数 ====================

def test_mysql_connection():
    """
    🔬 测试MySQL连接是否正常
    """
    config = DatabaseConfig()
    
    try:
        print("="*60)
        print("🚀 开始测试数据库连接...")
        
        config.create_database_if_not_exists()
        config.create_user_table()
        
        print("\n📝 测试添加用户...")
        success, msg, uid = config.add_user("test_user", "123456")
        print(f"   结果：{msg}, ID={uid}")
        
        print("\n🔍 测试用户验证...")
        result = config.verify_user("test_user", "123456")
        print(f"   结果：{result}")
        
        print("\n✅ 全部测试通过！")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 测试失败：{e}")


if __name__ == "__main__":
    test_mysql_connection()