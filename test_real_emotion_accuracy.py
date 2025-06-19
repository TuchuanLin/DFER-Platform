#!/usr/bin/env python3
"""
深度测试表情识别准确性
分析模型在不同情况下的表现
"""

import sys
import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import List, Dict, Any

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from fer_platform.models.improved_m3d_wrapper import ImprovedM3DInferenceWrapper
from fer_platform.config import config

def create_diverse_test_frames():
    """创建更多样化的测试帧来模拟不同表情"""
    test_cases = {}
    
    # 1. 基础测试帧（中性）
    base_frame = np.zeros((112, 112, 3), dtype=np.uint8)
    cv2.circle(base_frame, (56, 56), 40, (128, 128, 128), -1)  # 脸部
    cv2.circle(base_frame, (45, 45), 5, (255, 255, 255), -1)   # 左眼
    cv2.circle(base_frame, (67, 45), 5, (255, 255, 255), -1)   # 右眼
    cv2.ellipse(base_frame, (56, 70), (10, 5), 0, 0, 180, (255, 255, 255), 2)  # 中性嘴巴
    test_cases["中性"] = [base_frame.copy() for _ in range(16)]
    
    # 2. 开心表情（上扬的嘴巴）
    happy_frame = base_frame.copy()
    cv2.ellipse(happy_frame, (56, 70), (15, 8), 0, 0, 180, (255, 255, 255), 2)  # 微笑
    cv2.ellipse(happy_frame, (45, 50), (3, 2), 0, 0, 180, (200, 200, 200), 1)  # 笑眼
    cv2.ellipse(happy_frame, (67, 50), (3, 2), 0, 0, 180, (200, 200, 200), 1)  # 笑眼
    test_cases["开心"] = [happy_frame.copy() for _ in range(16)]
    
    # 3. 愤怒表情（皱眉）
    angry_frame = base_frame.copy()
    cv2.line(angry_frame, (40, 40), (50, 35), (200, 200, 200), 2)  # 左眉毛下压
    cv2.line(angry_frame, (62, 35), (72, 40), (200, 200, 200), 2)  # 右眉毛下压
    cv2.ellipse(angry_frame, (56, 75), (12, 6), 0, 180, 360, (255, 255, 255), 2)  # 下压的嘴
    test_cases["愤怒"] = [angry_frame.copy() for _ in range(16)]
    
    # 4. 恐惧表情（大眼睛，张嘴）
    fear_frame = base_frame.copy()
    cv2.circle(fear_frame, (45, 45), 8, (255, 255, 255), -1)   # 大左眼
    cv2.circle(fear_frame, (67, 45), 8, (255, 255, 255), -1)   # 大右眼
    cv2.circle(fear_frame, (45, 45), 3, (0, 0, 0), -1)         # 瞳孔
    cv2.circle(fear_frame, (67, 45), 3, (0, 0, 0), -1)         # 瞳孔
    cv2.ellipse(fear_frame, (56, 75), (8, 12), 90, 0, 180, (255, 255, 255), 2)  # 张开的嘴
    test_cases["恐惧"] = [fear_frame.copy() for _ in range(16)]
    
    # 5. 悲伤表情（下垂的眉毛和嘴巴）
    sad_frame = base_frame.copy()
    cv2.line(sad_frame, (40, 35), (50, 40), (200, 200, 200), 2)  # 下垂眉毛
    cv2.line(sad_frame, (62, 40), (72, 35), (200, 200, 200), 2)  # 下垂眉毛
    cv2.ellipse(sad_frame, (56, 75), (10, 5), 0, 180, 360, (255, 255, 255), 2)  # 下垂嘴巴
    test_cases["悲伤"] = [sad_frame.copy() for _ in range(16)]
    
    # 6. 惊讶表情（圆眼睛，圆嘴巴）
    surprise_frame = base_frame.copy()
    cv2.circle(surprise_frame, (45, 45), 7, (255, 255, 255), -1)   # 圆眼睛
    cv2.circle(surprise_frame, (67, 45), 7, (255, 255, 255), -1)   # 圆眼睛
    cv2.circle(surprise_frame, (45, 45), 2, (0, 0, 0), -1)         # 瞳孔
    cv2.circle(surprise_frame, (67, 45), 2, (0, 0, 0), -1)         # 瞳孔
    cv2.circle(surprise_frame, (56, 75), 6, (255, 255, 255), 2)    # 圆嘴巴
    test_cases["惊讶"] = [surprise_frame.copy() for _ in range(16)]
    
    return test_cases

def analyze_model_performance():
    """分析模型在不同情况下的表现"""
    print("🔍 深度分析表情识别准确性")
    print("=" * 60)
    
    # 初始化模型
    print("初始化模型...")
    wrapper = ImprovedM3DInferenceWrapper()
    
    # 获取模型信息
    model_info = wrapper.get_model_info()
    print(f"模型信息:")
    print(f"  名称: {model_info['model_name']}")
    print(f"  版本: {model_info['version']}")
    print(f"  设备: {model_info['device']}")
    print(f"  预训练模型: {'已加载' if model_info['pretrained_info']['has_pretrained'] else '未加载'}")
    print()
    
    # 创建测试用例
    print("创建测试用例...")
    test_cases = create_diverse_test_frames()
    
    results = {}
    print("=" * 60)
    print("测试结果:")
    print("=" * 60)
    
    for emotion_name, frames in test_cases.items():
        print(f"\n🎭 测试 {emotion_name} 表情:")
        
        # 进行预测
        result = wrapper.predict(frames)
        
        predicted_emotion = result['predicted_label']
        confidence = result['confidence']
        
        # 计算准确性
        is_correct = predicted_emotion == emotion_name
        accuracy_symbol = "✅" if is_correct else "❌"
        
        print(f"  预期: {emotion_name}")
        print(f"  预测: {predicted_emotion} {accuracy_symbol}")
        print(f"  置信度: {confidence:.4f}")
        
        # 显示前3个概率
        probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
        print(f"  前3概率:")
        for i, (emotion, prob) in enumerate(probs[:3]):
            print(f"    {i+1}. {emotion}: {prob:.4f}")
        
        # 保存结果
        results[emotion_name] = {
            'predicted': predicted_emotion,
            'confidence': confidence,
            'correct': is_correct,
            'probabilities': result['probabilities']
        }
        
        if 'improvement_info' in result:
            print(f"  改进信息:")
            print(f"    原始预测: {result['improvement_info']['original_prediction']}")
            print(f"    原始置信度: {result['improvement_info']['original_confidence']:.4f}")
            print(f"    校正应用: {'是' if result['improvement_info']['correction_applied'] else '否'}")
    
    # 计算总体准确率
    correct_predictions = sum(1 for r in results.values() if r['correct'])
    total_predictions = len(results)
    overall_accuracy = correct_predictions / total_predictions
    
    print("\n" + "=" * 60)
    print("📊 总体统计:")
    print("=" * 60)
    print(f"总测试用例: {total_predictions}")
    print(f"正确预测: {correct_predictions}")
    print(f"总体准确率: {overall_accuracy:.2%}")
    
    # 分析问题
    print("\n🔍 问题分析:")
    incorrect_cases = [(name, data) for name, data in results.items() if not data['correct']]
    
    if incorrect_cases:
        print("错误预测案例:")
        for emotion_name, data in incorrect_cases:
            print(f"  • {emotion_name} → {data['predicted']} (置信度: {data['confidence']:.4f})")
    else:
        print("✅ 所有测试用例都预测正确！")
    
    return results

def test_bias_correction_effectiveness():
    """测试偏差校正的有效性"""
    print("\n" + "=" * 60)
    print("🎛️ 测试偏差校正有效性")
    print("=" * 60)
    
    # 创建测试帧
    test_frames = create_diverse_test_frames()["中性"]  # 使用中性表情作为基准
    
    # 测试不同的偏差校正配置
    configurations = [
        {
            'name': '无偏差校正',
            'enable_bias_correction': False,
            'enable_confidence_filtering': False
        },
        {
            'name': '启用偏差校正',
            'enable_bias_correction': True,
            'enable_confidence_filtering': False
        },
        {
            'name': '偏差校正+置信度过滤',
            'enable_bias_correction': True,
            'enable_confidence_filtering': True
        }
    ]
    
    for config_test in configurations:
        print(f"\n🔧 配置: {config_test['name']}")
        
        wrapper = ImprovedM3DInferenceWrapper(
            enable_bias_correction=config_test['enable_bias_correction'],
            enable_confidence_filtering=config_test['enable_confidence_filtering']
        )
        
        result = wrapper.predict(test_frames)
        
        print(f"  预测: {result['predicted_label']}")
        print(f"  置信度: {result['confidence']:.4f}")
        
        # 显示概率分布
        probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
        print(f"  概率分布:")
        for emotion, prob in probs:
            print(f"    {emotion}: {prob:.4f}")

def suggest_improvements():
    """提供改进建议"""
    print("\n" + "=" * 60)
    print("💡 改进建议")
    print("=" * 60)
    
    suggestions = [
        "1. 数据预处理改进:",
        "   • 增强人脸检测和对齐",
        "   • 改进图像归一化方法",
        "   • 添加数据增强技术",
        "",
        "2. 模型架构优化:",
        "   • 使用更先进的骨干网络",
        "   • 添加注意力机制",
        "   • 改进时序建模",
        "",
        "3. 训练策略改进:",
        "   • 使用更大规模的训练数据",
        "   • 应用类别平衡技术",
        "   • 实施难样本挖掘",
        "",
        "4. 偏差校正优化:",
        "   • 基于验证集动态调整权重",
        "   • 实现自适应温度缩放",
        "   • 添加不确定性估计",
        "",
        "5. 实际应用优化:",
        "   • 添加人脸质量评估",
        "   • 实现多模态融合",
        "   • 增加上下文信息利用"
    ]
    
    for suggestion in suggestions:
        print(suggestion)

def main():
    """主函数"""
    print("🚀 开始表情识别准确性深度分析")
    print()
    
    try:
        # 1. 分析模型性能
        results = analyze_model_performance()
        
        # 2. 测试偏差校正
        test_bias_correction_effectiveness()
        
        # 3. 提供改进建议
        suggest_improvements()
        
        print("\n" + "=" * 60)
        print("✅ 分析完成")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 