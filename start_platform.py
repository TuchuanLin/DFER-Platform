#!/usr/bin/env python3
"""
动态表情识别平台启动脚本
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    if sys.version_info < (3, 8):
        logger.error("需要Python 3.8或更高版本")
        return False
    logger.info(f"Python版本: {sys.version}")
    return True

def create_directories():
    directories = ['logs', 'models', 'data/uploads', 'data/results']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"创建目录: {directory}")

def start_server():
    logger.info("启动开发服务器...")
    logger.info("访问地址: http://localhost:8000")
    try:
        subprocess.run([sys.executable, "-m", "uvicorn", "fer_platform.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])
    except KeyboardInterrupt:
        logger.info("服务器已停止")

def main():
    print("=" * 60)
    print("    动态表情识别平台 (M3D-based FER Platform)")
    print("=" * 60)
    
    if not check_python_version():
        return 1
    
    create_directories()
    start_server()
    return 0

if __name__ == "__main__":
    sys.exit(main()) 