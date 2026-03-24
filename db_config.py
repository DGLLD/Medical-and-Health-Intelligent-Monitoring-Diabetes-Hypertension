#!/usr/bin/env python3
"""
📋 文件名称: db_config.py
🎯 功能描述: 数据库连接与用户操作模块
"""

import pymysql
import os
from datetime import datetime


class DatabaseConfig:
    """
    🔧 数据库配置类 - 管理MySQL连接和用户操作
    """
    
    def __init__(self, host=None, port=None, user=None, 
                 password=None, database=None, charset='utf8mb4'):
        """
        【构造方法】初始化数据库连接参数
        支持环境变量覆盖，便于Docker部署
        """
        self.config = {
            'host': host or os.getenv('DB_HOST', '127.0.0.1'),
            'port': port or int(os.getenv('DB_PORT', 3306)),
            'user': user or os.getenv('DB_USER', 'root'),
            'password': password or os.getenv('DB_PASSWORD', '111111'),
            'database': database or os.getenv('DB_NAME', 'test_db'),
            'charset': charset,
            'cursorclass': pymysql.cursors.DictCursor
        }
    
    def get_connection(self):
        """
        🔗 获取数据库连接对象
        """
        conn = pymysql.connect(**self.config)
        return conn
    
    def create_database_if_not_exists(self, db_name=None):
        """
        🗂️ 检查并创建数据库（如果不存在）
        """
        db_name = db_name or self.config['database']
        conn = None
        cursor = None
        
        try:
            temp_config = {k: v for k, v in self.config.items() if k != 'database'}
            conn = pymysql.connect(**temp_config)
            cursor = conn.cursor()
            
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
        """
        conn = None
        cursor = None
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS `{}`.`users` (
                `id` INT AUTO_INCREMENT PRIMARY KEY COMMENT '用户ID',
                `username` VARCHAR(50) NOT NULL UNIQUE COMMENT '用户账号',
                `password` VARCHAR(128) NOT NULL COMMENT '密码哈希值',
                `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
                `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
                INDEX idx_username (`username`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """.format(self.config['database'])
            
            cursor.execute(create_table_sql)
            conn.commit()
            print("✅ 用户表 'users' 创建成功")
            return True
            
        except Exception as e:
            print(f"❌ 创建用户表失败：{e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if cursor: cursor.close()
            if conn: conn.close()
    
    def user_exists(self, username):
        """
        ✅ 检查账号是否存在
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
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            select_sql = "SELECT id, username, password FROM `users` WHERE username = %s"
            cursor.execute(select_sql, (username,))
            user = cursor.fetchone()
            
            if not user:
                return {'success': False, 'user_id': None, 'message': '账号不存在'}
            
            stored_hash = user['password']
            
            if stored_hash == plain_password:
                print(f"✅ 用户验证成功：{username}")
                return {'success': True, 'user_id': user['id'], 'message': '登录成功'}
            else:
                return {'success': False, 'user_id': None, 'message': '密码错误'}
            
        except Exception as e:
            return {'success': False, 'user_id': None, 'message': f'验证错误：{str(e)}'}
        finally:
            cursor.close()
            conn.close()


if __name__ == "__main__":
    config = DatabaseConfig()
    config.create_database_if_not_exists()
    config.create_user_table()