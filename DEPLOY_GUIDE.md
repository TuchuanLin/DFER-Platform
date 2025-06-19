# DFER-Platform 部署指南

## 📋 更新内容总结

### 🔄 模型名称更新
- 将所有M3D模型引用更新为**MICACL**（Multi-Instance Contrastive Active Learning）
- 更新了API响应中的模型名称为"MICACL-FER"
- 更新了网页界面中的模型描述

### 🎨 网页界面优化
- **新增现代化SVG图片**: 创建了`fer_platform/static/images/micacl-hero.svg`，展示MICACL动态表情识别概念
- **首页图片更新**: 将首页展示图片从`emotion-demo.svg`更换为新的`micacl-hero.svg`
- **美观度提升**: 新图片包含动态效果、神经网络可视化和现代化设计

### 📊 关于页面更新
- **准确率调整**: 将平均准确率从95%调整为**69%**
- **研发团队**: 更新为**hfut**
- **技术支持**: 更新为**hfut**
- **模型架构**: 全面更新为MICACL相关描述

### 🔧 代码更新
- 将`ImprovedM3DInferenceWrapper`类重命名为`MICACLInferenceWrapper`
- 更新所有相关的导入和引用
- 保持原有功能不变，仅更新命名

## 🚀 上传到GitHub指南

### 前提条件
1. 确保已安装Git
2. 确保已配置SSH密钥访问GitHub
3. 确保GitHub仓库`git@github.com:TuchuanLin/DFER-Platform.git`已创建

### 方式一：使用提供的脚本

#### Windows用户：
```bash
upload_to_github.bat
```

#### Linux/Mac用户：
```bash
chmod +x upload_to_github.sh
./upload_to_github.sh
```

### 方式二：手动上传

```bash
# 1. 初始化Git仓库
git init

# 2. 添加所有文件
git add .

# 3. 创建初始提交
git commit -m "Initial commit: MICACL Dynamic Facial Expression Recognition Platform"

# 4. 添加远程仓库
git remote add origin git@github.com:TuchuanLin/DFER-Platform.git

# 5. 推送到GitHub
git push -u origin main
```

## 📁 文件结构说明

```
DFER-Platform/
├── .gitignore                     # Git忽略文件规则
├── README.md                      # 项目说明文档
├── DEPLOY_GUIDE.md               # 本部署指南
├── requirements.txt              # Python依赖
├── start_platform.py            # 平台启动脚本
├── upload_to_github.bat         # Windows上传脚本
├── upload_to_github.sh          # Linux/Mac上传脚本
├── fer_platform/                 # 主要平台代码
│   ├── static/images/
│   │   └── micacl-hero.svg      # 新的首页展示图片
│   ├── templates/
│   │   ├── index.html           # 更新的首页模板
│   │   └── about.html           # 更新的关于页面
│   ├── models/
│   │   └── improved_m3d_wrapper.py  # 重命名为MICACL
│   └── api/
│       └── endpoints.py         # 更新的API接口
├── data/                        # 数据目录
│   ├── uploads/.gitkeep        # 保持目录结构
│   └── results/.gitkeep        # 保持目录结构
└── models/                     # 模型文件目录
```

## ⚠️ 注意事项

1. **大文件处理**: `.gitignore`已配置忽略大型模型文件（.pth, .pt, .pkl）
2. **敏感数据**: 确保不包含任何敏感配置或密钥
3. **依赖管理**: 确保`requirements.txt`包含所有必要依赖
4. **文档同步**: 保持README.md与实际功能同步

## 🔧 后续维护

1. **定期更新**: 根据模型改进定期更新文档和代码
2. **版本控制**: 使用Git标签管理版本发布
3. **问题跟踪**: 利用GitHub Issues管理bug和功能请求
4. **文档维护**: 保持技术文档与代码同步

## 📞 技术支持

如在部署过程中遇到问题，请：
1. 检查Git和SSH配置
2. 确认GitHub仓库访问权限
3. 查看终端错误信息
4. 联系技术支持团队：hfut 