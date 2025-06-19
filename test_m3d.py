#!/usr/bin/env python3
"""
测试M3D模型推理功能
"""

import numpy as np
import torch
import cv2
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from fer_platform.models.m3d_wrapper import M3DInferenceWrapper

def test_m3d_inference():
    """测试M3D推理功能"""
    print("=" * 50)
    print("M3D模型推理测试")
    print("=" * 50)
    
    try:
        # 初始化模型
        print("1. 初始化模型...")
        wrapper = M3DInferenceWrapper(device="cpu")  # 使用CPU避免CUDA问题
        print("✓ 模型初始化成功")
        
        # 创建测试帧
        print("\n2. 创建测试帧...")
        test_frames = []
        for i in range(16):
            # 创建随机图像 (112, 112, 3)
            frame = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            test_frames.append(frame)
        print(f"✓ 创建了 {len(test_frames)} 个测试帧")
        
        # 测试预处理
        print("\n3. 测试预处理...")
        tensor = wrapper.preprocess_frames(test_frames)
        print(f"✓ 预处理成功，输出形状: {tensor.shape}")
        
        # 测试推理
        print("\n4. 测试推理...")
        result = wrapper.predict(test_frames)
        print("✓ 推理成功")
        print(f"  - 预测表情: {result['predicted_label']}")
        print(f"  - 置信度: {result['confidence']:.3f}")
        print(f"  - 预测类别: {result['predicted_class']}")
        
        # 显示概率分布
        print("\n5. 概率分布:")
        for emotion, prob in result['probabilities'].items():
            print(f"  - {emotion}: {prob:.3f}")
        
        print("\n" + "=" * 50)
        print("✓ 所有测试通过！模型运行正常")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_single_frame():
    """测试单帧处理"""
    print("\n测试单帧处理...")
    
    try:
        wrapper = M3DInferenceWrapper(device="cpu")
        
        # 创建单个测试帧
        single_frame = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        frames = [single_frame]  # 只有一帧
        
        result = wrapper.predict(frames)
        print(f"✓ 单帧处理成功: {result['predicted_label']}")
        
        return True
    except Exception as e:
        print(f"❌ 单帧处理失败: {e}")
        return False

def test_grayscale_frame():
    """测试灰度图处理"""
    print("\n测试灰度图处理...")
    
    try:
        wrapper = M3DInferenceWrapper(device="cpu")
        
        # 创建灰度图
        gray_frame = np.random.randint(0, 255, (112, 112), dtype=np.uint8)
        frames = [gray_frame]
        
        result = wrapper.predict(frames)
        print(f"✓ 灰度图处理成功: {result['predicted_label']}")
        
        return True
    except Exception as e:
        print(f"❌ 灰度图处理失败: {e}")
        return False

if __name__ == "__main__":
    success = True
    
    # 运行所有测试
    success &= test_m3d_inference()
    success &= test_single_frame()
    success &= test_grayscale_frame()
    
    if success:
        print("\n🎉 所有测试都通过了！")
        sys.exit(0)
    else:
        print("\n❌ 有测试失败")
        sys.exit(1) 