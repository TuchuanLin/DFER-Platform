# 动态表情识别平台 - 项目结构说明

## 项目概述

本项目是基于M3D模型的实时动态表情识别平台，将原有的学术研究代码重构为可部署的Web应用平台。

## 目录结构

```
ICMEDFER/
├── README.md                    # 项目说明文档
├── requirements.txt             # Python依赖包列表
├── start_platform.py           # 平台启动脚本
├── Dockerfile                   # Docker镜像构建文件
├── docker-compose.yml          # Docker编排配置
│
├── platform/                   # 平台核心代码
│   ├── app.py                  # FastAPI主应用
│   ├── api/                    # API接口层
│   │   └── endpoints.py        # REST API和WebSocket接口
│   ├── core/                   # 核心业务逻辑
│   │   ├── video_processor.py  # 视频处理器
│   │   └── task_manager.py     # 任务管理器
│   ├── models/                 # 模型封装层
│   │   └── m3d_wrapper.py      # M3D模型推理包装器
│   ├── static/                 # 静态资源
│   │   ├── css/
│   │   │   └── style.css       # 样式文件
│   │   └── js/
│   │       └── upload.js       # 前端JavaScript
│   └── templates/              # HTML模板
│       ├── index.html          # 主页
│       └── upload.html         # 上传页面
│
├── core_model/                 # M3D核心模型 (重构后)
│   ├── __init__.py
│   ├── models/                 # 模型定义
│   │   ├── __init__.py
│   │   └── M3D.py             # M3D模型架构
│   ├── datasets/              # 数据处理
│   │   ├── __init__.py
│   │   ├── dataset_DFEW.py    # DFEW数据集处理
│   │   └── video_transform.py # 视频变换
│   ├── utils/                 # 工具函数
│   │   ├── __init__.py
│   │   └── utils.py          # 通用工具
│   └── options.py            # 配置选项
│
├── tests/                     # 测试用例
├── docs/                      # 文档目录
├── docker/                    # Docker相关配置
└── attribute/                 # 原始代码(保留)
    └── M3DFEL/               # 原始M3D实现
```

## 核心组件说明

### 1. 平台层 (platform/)

**FastAPI应用 (`app.py`)**
- 主应用入口，配置路由、中间件、静态文件服务
- 提供Web界面路由和API路由
- 集成日志系统和错误处理

**API接口 (`api/endpoints.py`)**
- REST API: 视频上传、结果查询、任务管理
- WebSocket API: 实时摄像头识别
- 异步任务处理

**视频处理器 (`core/video_processor.py`)**
- 支持多种视频格式解析
- 帧提取和预处理
- 摄像头实时捕获

**任务管理器 (`core/task_manager.py`)**
- 异步任务状态管理
- 结果缓存和清理
- 线程安全的任务操作

**模型包装器 (`models/m3d_wrapper.py`)**
- M3D模型的高级封装
- 统一的推理接口
- 设备管理和性能优化

### 2. 核心模型层 (core_model/)

**M3D模型 (`models/M3D.py`)**
- 原始M3D架构的重构版本
- 支持图卷积和BiLSTM的混合架构
- 多头自注意力机制

**数据处理 (`datasets/`)**
- 视频数据加载和预处理
- 数据增强和变换
- 批处理支持

**工具函数 (`utils/`)**
- 学习率调度器
- 动态多实例标准化
- 通用工具函数

### 3. 前端界面

**主页 (`templates/index.html`)**
- 响应式设计
- 功能介绍和导航
- 现代化UI/UX

**上传页面 (`templates/upload.html`)**
- 拖拽上传支持
- 实时进度显示
- 结果可视化

**样式系统 (`static/css/style.css`)**
- 自定义CSS变量
- 动画效果
- 移动端适配

**交互逻辑 (`static/js/upload.js`)**
- 文件验证和上传
- 异步结果轮询
- 动态图表展示

## 主要功能

### 1. 视频文件识别
- 支持多种视频格式 (MP4, AVI, MOV, MKV, WMV, FLV, WEBM)
- 自动帧提取和预处理
- 异步处理和进度跟踪
- 详细的识别结果展示

### 2. 实时摄像头识别
- WebSocket实时通信
- 摄像头流处理
- 实时表情分析
- 低延迟结果反馈

### 3. 结果可视化
- 7种表情类别识别
- 置信度评分
- 概率分布图表
- 历史记录管理

### 4. 系统监控
- API健康检查
- 任务状态监控
- 性能指标统计
- 错误日志记录

## 部署方式

### 1. 开发环境
```bash
# 安装依赖
pip install -r requirements.txt

# 启动开发服务器
python start_platform.py
```

### 2. Docker部署
```bash
# 构建镜像
docker build -t fer-platform .

# 运行容器
docker run -p 8000:8000 fer-platform
```

### 3. Docker Compose
```bash
# 启动完整服务栈
docker-compose up -d
```

## 技术特性

### 后端技术栈
- **Web框架**: FastAPI (高性能异步框架)
- **深度学习**: PyTorch + M3D模型
- **视频处理**: OpenCV
- **任务队列**: 内存队列 + Redis (可选)
- **API文档**: 自动生成的OpenAPI文档

### 前端技术栈
- **UI框架**: Bootstrap 5
- **图标**: Font Awesome
- **图表**: Chart.js
- **交互**: 原生JavaScript (ES6+)

### 基础设施
- **容器化**: Docker + Docker Compose
- **反向代理**: Nginx
- **监控**: Prometheus + Grafana (可选)
- **缓存**: Redis

## 性能优化

1. **模型推理优化**
   - GPU加速支持
   - 批处理推理
   - 模型权重缓存

2. **视频处理优化**
   - 异步文件上传
   - 流式视频处理
   - 内存管理优化

3. **Web服务优化**
   - 静态文件缓存
   - gzip压缩
   - CDN支持

4. **用户体验优化**
   - 实时进度反馈
   - 响应式设计
   - 错误处理机制

## 扩展性

### 水平扩展
- 多实例部署支持
- 负载均衡配置
- 分布式任务处理

### 功能扩展
- 批量视频处理
- 用户管理系统
- 数据统计分析
- 模型版本管理

### 集成扩展
- REST API集成
- 第三方服务集成
- 数据库持久化
- 云存储支持

## 安全考虑

1. **文件上传安全**
   - 文件类型验证
   - 文件大小限制
   - 病毒扫描 (可扩展)

2. **API安全**
   - 请求频率限制
   - 输入验证
   - 错误信息过滤

3. **部署安全**
   - HTTPS支持
   - 环境变量管理
   - 容器安全配置

这个平台为M3D模型提供了完整的Web服务封装，便于部署、使用和维护。 