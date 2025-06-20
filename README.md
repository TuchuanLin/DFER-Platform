# DFER-Platform: MICACL动态表情识别平台

[![GitHub](https://img.shields.io/github/license/TuchuanLin/DFER-Platform)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-orange.svg)](https://pytorch.org/)

基于MICACL（Multi-Instance Contrastive Active Learning）深度学习模型的先进动态表情识别系统，提供实时和批量视频表情分析服务。

## 🌟 主要特性

- **🎯 高精度识别**: 基于MICACL架构，平均准确率达69%
- **⚡ 实时处理**: 支持摄像头实时表情识别
- **📹 批量处理**: 支持多种视频格式上传分析
- **🎨 可视化界面**: 现代化Web界面，直观展示识别结果
- **🔌 API接口**: 完整的RESTful API支持第三方集成
- **📊 详细分析**: 提供置信度评分和概率分布

## 🎭 支持的表情类别

- 😊 快乐 (Happy)
- 😢 悲伤 (Sad)
- 😠 愤怒 (Angry)
- 😲 惊讶 (Surprise)
- 😨 恐惧 (Fear)
- 🤢 厌恶 (Disgust)
- 😐 中性 (Neutral)

## 🏗️ 系统架构

```
DFER-Platform/
├── fer_platform/           # 主要平台代码
│   ├── api/               # API接口
│   ├── core/              # 核心处理模块
│   ├── models/            # 模型包装器
│   ├── static/            # 静态资源
│   └── templates/         # HTML模板
├── core_model/            # 核心模型代码
├── models/                # 预训练模型文件
├── data/                  # 数据目录
├── docs/                  # 文档
└── tests/                 # 测试代码
```

## 🎬 演示视频

观看平台演示视频了解完整功能：

[![MICACL表情识别平台演示](https://img.shields.io/badge/演示视频-观看-red.svg)](demo_video.mp4)

演示内容包括：
- 实时摄像头表情识别
- 视频上传和批量处理
- 结果可视化和分析
- API接口使用示例

> **注意**: 演示视频使用Git LFS存储，首次克隆仓库时请确保安装了Git LFS。

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.8+
- CUDA支持（推荐）

### 安装依赖

```bash
git clone https://github.com/TuchuanLin/DFER-Platform.git
cd DFER-Platform
pip install -r requirements.txt
```

### 启动平台

```bash
python start_platform.py
```

访问 `http://localhost:8000` 开始使用平台。

## 📖 使用指南

### 1. 视频上传识别
- 访问"视频上传"页面
- 选择视频文件（支持MP4、AVI、MOV等格式）
- 上传后系统自动分析
- 在"结果查看"页面查看识别结果

### 2. 实时识别
- 进入"实时识别"页面
- 允许摄像头访问
- 实时查看表情识别结果

### 3. API使用
访问 `/docs` 查看完整API文档。

主要接口：
- `POST /api/upload` - 上传视频
- `GET /api/result/{task_id}` - 获取结果
- `WebSocket /api/realtime` - 实时识别
- `GET /api/model/info` - 模型信息

## 🔬 技术详情

### MICACL模型架构
MICACL（Multi-Instance Contrastive Active Learning）是专为动态表情识别设计的深度学习架构，具有以下特点：

- **多实例学习**: 有效处理视频序列中的时间依赖关系
- **对比学习**: 提高特征表示的判别能力
- **主动学习**: 优化训练样本选择策略

### 技术栈
- **深度学习**: PyTorch, MICACL架构
- **后端**: FastAPI, 异步处理
- **前端**: Bootstrap, WebSocket实时通信
- **部署**: Docker支持

## 📊 性能指标

| 指标 | 数值 |
|------|------|
| 平均准确率 | 69% |
| 帧序列长度 | 16帧 |
| 表情类别 | 7种 |
| 实时处理速度 | 30fps |

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目。

## 📄 许可证

本项目采用MIT许可证。详见[LICENSE](LICENSE)文件。

## 👥 开发团队

- **研发团队**: hfut
- **技术支持**: hfut

## 📧 联系我们

如有问题或建议，请提交Issue或联系开发团队。

---

**⚠️ 免责声明**: 本平台仅供学术研究和技术交流使用，请遵守相关法律法规和伦理规范。 