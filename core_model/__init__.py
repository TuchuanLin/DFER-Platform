"""
M3D核心模型包
包含动态表情识别的核心模型和数据处理功能
"""

from .models.M3D import M3DFEL
from .utils.utils import build_scheduler, DMIN

__all__ = ['M3DFEL', 'build_scheduler', 'DMIN']
__version__ = '1.0.0' 