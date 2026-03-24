#!/bin/bash
set -e

echo "=========================================="
echo "   医疗健康风险评估系统 - Docker启动"
echo "=========================================="

echo "等待MySQL数据库连接..."
sleep 5

# 检查模型是否存在，不存在则训练
if [ ! -f "models/hypertension_model.pkl" ] || [ ! -f "models/diabetes_model_general.pkl" ] || [ ! -f "models/diabetes_model_pima.pkl" ]; then
    echo "模型文件不存在，开始训练模型..."
    python src/train_model.py
    echo "模型训练完成！"
else
    echo "模型文件已存在，跳过训练..."
fi

echo "启动Flask应用..."
echo "访问地址: http://localhost:${APP_PORT:-5000}"
echo "=========================================="

exec python app.py