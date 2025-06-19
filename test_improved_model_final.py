#!/usr/bin/env python3
"""
最终测试改进后的M3D模型
验证预训练权重是否正常工作
"""

import sys
import os
import numpy as np
import cv2
import torch
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from fer_platform.models.improved_m3d_wrapper import ImprovedM3DInferenceWrapper
from fer_platform.config import config

def create_test_frames():
    """创建测试用的视频帧"""
    # 创建一个简单的测试图像
    test_frame = np.zeros((112, 112, 3), dtype=np.uint8)
    
    # 绘制一个简单的脸部形状（用于测试）
    cv2.circle(test_frame, (56, 56), 40, (128, 128, 128), -1)  # 脸部
    cv2.circle(test_frame, (45, 45), 5, (255, 255, 255), -1)   # 左眼
    cv2.circle(test_frame, (67, 45), 5, (255, 255, 255), -1)   # 右眼
    cv2.ellipse(test_frame, (56, 70), (15, 8), 0, 0, 180, (255, 255, 255), 2)  # 嘴巴
    
    # 复制16帧
    frames = [test_frame.copy() for _ in range(16)]
    return frames

def test_model_loading():
    """测试模型加载状态"""
    print("=" * 60)
    print("测试改进后的M3D模型")
    print("=" * 60)
    
    # 检查配置
    print(f"项目根目录: {config.PROJECT_ROOT}")
    print(f"模型目录: {config.MODEL_CONFIG['model_dir']}")
    print(f"模型路径: {config.get_model_path()}")
    print(f"有预训练模型: {config.has_pretrained_model()}")
    
    if config.has_pretrained_model():
        model_path = Path(config.get_model_path())
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"模型文件大小: {size_mb:.1f} MB")
        else:
            print("⚠️ 模型路径不存在")
    print()

def test_model_prediction():
    """测试模型预测"""
    print("初始化改进的M3D模型...")
    
    try:
        # 创建模型实例
        model = ImprovedM3DInferenceWrapper(
            enable_bias_correction=True,
            enable_confidence_filtering=True,
            confidence_threshold=0.3
        )
        
        print(f"✅ 模型初始化成功")
        print(f"   设备: {model.device}")
        print(f"   偏差校正: {'启用' if model.enable_bias_correction else '禁用'}")
        print(f"   置信度过滤: {'启用' if model.enable_confidence_filtering else '禁用'}")
        print()
        
        # 获取模型信息
        model_info = model.get_model_info()
        print("模型信息:")
        print(f"  模型名称: {model_info['model_name']}")
        print(f"  版本: {model_info['version']}")
        print(f"  类别数: {model_info['num_classes']}")
        print(f"  输入形状: {model_info['input_shape']}")
        print(f"  预训练模型: {'已加载' if model_info['pretrained_info']['has_pretrained'] else '未加载'}")
        print()
        
        # 测试预测
        print("进行测试预测...")
        test_frames = create_test_frames()
        
        # 进行预测
        result = model.predict(test_frames)
        
        print("预测结果:")
        print(f"  预测表情: {result['predicted_label']}")
        print(f"  置信度: {result['confidence']:.4f}")
        print("  概率分布:")
        for emotion, prob in result['probabilities'].items():
            print(f"    {emotion}: {prob:.4f}")
        
        # 显示改进信息
        if 'improvement_info' in result:
            info = result['improvement_info']
            print("\n改进信息:")
            print(f"  原始预测: {info['original_prediction']} (置信度: {info['original_confidence']:.4f})")
            print(f"  校正应用: {'是' if info['correction_applied'] else '否'}")
            print(f"  置信度过滤: {'是' if info['confidence_filtering_applied'] else '否'}")
        
        # 模型状态信息
        model_info = result.get('model_info', {})
        print("\n模型状态:")
        print(f"  预训练权重: {'已加载' if model_info.get('has_pretrained', False) else '随机初始化'}")
        print(f"  设备: {model_info.get('device', 'unknown')}")
        print(f"  偏差校正: {'启用' if model_info.get('bias_correction_enabled', False) else '禁用'}")
        
        if 'error' in result:
            print(f"\n⚠️ 警告: {result['error']}")
        
        print("\n✅ 测试完成")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_bias_correction():
    """测试偏差校正功能"""
    print("\n" + "=" * 60)
    print("测试偏差校正功能")
    print("=" * 60)
    
    try:
        # 测试不同配置
        configs = [
            {"enable_bias_correction": False, "name": "原始模型"},
            {"enable_bias_correction": True, "name": "启用偏差校正"}
        ]
        
        test_frames = create_test_frames()
        
        for config_test in configs:
            print(f"\n测试配置: {config_test['name']}")
            
            model = ImprovedM3DInferenceWrapper(
                enable_bias_correction=config_test['enable_bias_correction'],
                enable_confidence_filtering=True,
                confidence_threshold=0.3
            )
            
            result = model.predict(test_frames)
            
            print(f"  预测: {result['predicted_label']} (置信度: {result['confidence']:.4f})")
            
            # 显示前三个最高概率
            sorted_probs = sorted(result['probabilities'].items(), 
                                key=lambda x: x[1], reverse=True)[:3]
            print("  前三概率:")
            for emotion, prob in sorted_probs:
                print(f"    {emotion}: {prob:.4f}")
        
        print("\n✅ 偏差校正测试完成")
        
    except Exception as e:
        print(f"❌ 偏差校正测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("开始最终模型测试...")
    
    # 测试模型加载
    test_model_loading()
    
    # 测试模型预测
    success = test_model_prediction()
    
    # 测试偏差校正
    if success:
        test_bias_correction()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    main() 