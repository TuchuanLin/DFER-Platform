"""
平台配置文件
"""
import os
from pathlib import Path

class Config:
    """平台配置类"""
    
    def __init__(self):
        # 项目根目录
        self.PROJECT_ROOT = Path(__file__).parent.parent
        
        # 模型配置
        self.MODEL_CONFIG = {
            # 默认模型路径（可以是None）
            'model_path': self._get_model_path(),
            
            # 支持的模型文件名列表（按优先级排序）
            'supported_model_files': [
                'model_best.pth',
                'm3d_checkpoint.pth', 
                'm3d_model.pth',
                'checkpoint.pth'
            ],
            
            # 模型目录
            'model_dir': self.PROJECT_ROOT / 'models',
            
            # 设备配置
            'device': 'auto',  # 'auto', 'cpu', 'cuda'
            
            # 模型参数
            'num_classes': 7,
            'num_frames': 16,
            'crop_size': 112
        }
        
        # 日志配置
        self.LOG_CONFIG = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    
    def _get_model_path(self):
        """自动查找可用的模型文件"""
        model_dir = self.PROJECT_ROOT / 'models'
        
        # 支持的模型文件名
        model_files = [
            'model_best.pth',
            'm3d_checkpoint.pth', 
            'm3d_model.pth',
            'checkpoint.pth'
        ]
        
        # 查找第一个存在的模型文件
        for model_file in model_files:
            model_path = model_dir / model_file
            if model_path.exists():
                return str(model_path)
        
        return None
    
    def get_model_path(self):
        """获取模型路径"""
        return self.MODEL_CONFIG['model_path']
    
    def has_pretrained_model(self):
        """检查是否有预训练模型"""
        return self.MODEL_CONFIG['model_path'] is not None
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'has_pretrained': self.has_pretrained_model(),
            'model_path': self.get_model_path(),
            'model_dir': str(self.MODEL_CONFIG['model_dir']),
            'supported_files': self.MODEL_CONFIG['supported_model_files']
        }

# 全局配置实例
config = Config() 