# Docker 部署指南

## 前置要求

- Docker Desktop 或 Docker Engine (版本 20.10+)
- Docker Compose (新版 Docker 已内置)

## 一键启动

### 方式一：使用默认配置

```bash
# 克隆项目
git clone <项目地址>
cd health-risk-predictor

# 启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f app

# 停止服务
docker-compose down