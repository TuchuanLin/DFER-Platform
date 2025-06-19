#!/usr/bin/env python3
"""
测试改进模型v2.0
验证新的偏差校正参数和置信度过滤效果
"""

import numpy as np
import cv2
import torch
from fer_platform.models.improved_m3d_wrapper import ImprovedM3DInferenceWrapper
from fer_platform.models.m3d_wrapper import M3DInferenceWrapper

def create_test_frames(emotion_type="neutral"):
    """创建测试帧"""
    # 创建一个简单的测试图像
    frame = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    
    # 添加一些模式以区分不同表情
    if emotion_type == "happy":
        # 添加白色区域模拟笑容
        frame[80:100, 40:70] = [255, 255, 255]
    elif emotion_type == "sad":
        # 添加蓝色区域
        frame[70:90, 40:70] = [0, 0, 255]
    elif emotion_type == "angry":
        # 添加红色区域
        frame[60:80, 35:75] = [255, 0, 0]
    
    return [frame.copy() for _ in range(16)]

def test_models():
    """测试模型比较"""
    print("初始化模型...")
    
    # 初始化原始模型
    original_model = M3DInferenceWrapper()
    
    # 初始化改进模型
    improved_model = ImprovedM3DInferenceWrapper()
    
    print(f"原始模型设备: {original_model.device}")
    print(f"改进模型设备: {improved_model.device}")
    print(f"改进模型配置: 偏差校正={improved_model.enable_bias_correction}, 置信度过滤={improved_model.enable_confidence_filtering}")
    print(f"置信度阈值: {improved_model.confidence_threshold}")
    print(f"温度参数: {improved_model.TEMPERATURE}")
    print("偏差校正权重:", improved_model.BIAS_CORRECTION_WEIGHTS)
    print()
    
    # 测试不同类型的输入
    test_cases = ["neutral", "happy", "sad", "angry"]
    
    results = []
    
    for test_case in test_cases:
        print(f"测试 {test_case} 表情:")
        print("-" * 50)
        
        frames = create_test_frames(test_case)
        
        # 原始模型预测
        original_result = original_model.predict(frames)
        print(f"原始模型预测: {original_result['predicted_label']} (置信度: {original_result['confidence']:.3f})")
        
        # 改进模型预测
        improved_result = improved_model.predict(frames)
        print(f"改进模型预测: {improved_result['predicted_label']} (置信度: {improved_result['confidence']:.3f})")
        
        # 显示改进信息
        if 'improvement_info' in improved_result:
            info = improved_result['improvement_info']
            print(f"原始预测: {info['original_prediction']} -> 校正后: {improved_result['predicted_label']}")
            print(f"是否应用校正: {info['correction_applied']}")
            print(f"是否应用置信度过滤: {info.get('confidence_filtering_applied', False)}")
        
        # 显示概率分布
        print("改进模型概率分布:")
        for emotion, prob in improved_result['probabilities'].items():
            print(f"  {emotion}: {prob:.3f}")
        
        results.append({
            'test_case': test_case,
            'original': original_result,
            'improved': improved_result
        })
        
        print()
    
    # 分析结果
    print("=" * 60)
    print("分析结果:")
    print("=" * 60)
    
    neutral_predictions = 0
    anger_fear_predictions = 0
    other_predictions = 0
    
    for result in results:
        improved_pred = result['improved']['predicted_label']
        
        if improved_pred == '中性':
            neutral_predictions += 1
        elif improved_pred in ['愤怒', '恐惧']:
            anger_fear_predictions += 1
        else:
            other_predictions += 1
    
    total = len(results)
    print(f"中性预测比例: {neutral_predictions}/{total} ({neutral_predictions/total:.1%})")
    print(f"愤怒/恐惧预测比例: {anger_fear_predictions}/{total} ({anger_fear_predictions/total:.1%})")
    print(f"其他表情预测比例: {other_predictions}/{total} ({other_predictions/total:.1%})")
    
    if neutral_predictions/total > 0.7:
        print("⚠️  警告: 中性预测比例过高，可能需要进一步调整参数")
    elif anger_fear_predictions/total > 0.5:
        print("⚠️  警告: 愤怒/恐惧预测比例仍然较高")
    else:
        print("✅ 预测分布相对均衡")
    
    return results

def test_parameter_adjustment():
    """测试参数调整功能"""
    print("\n" + "=" * 60)
    print("测试参数调整功能:")
    print("=" * 60)
    
    # 初始化改进模型
    model = ImprovedM3DInferenceWrapper()
    
    # 测试帧
    frames = create_test_frames("neutral")
    
    # 原始预测
    result1 = model.predict(frames)
    print(f"原始参数预测: {result1['predicted_label']} (置信度: {result1['confidence']:.3f})")
    
    # 调整参数
    print("\n调整参数: 降低中性权重到0.8，提高开心权重到1.5")
    config_result = model.configure_parameters(
        bias_correction_weights={4: 0.8, 3: 1.5},  # 中性权重0.8，开心权重1.5
        temperature=1.0  # 降低温度
    )
    print(f"参数调整结果: {config_result['status']}")
    
    # 新参数预测
    result2 = model.predict(frames)
    print(f"调整后预测: {result2['predicted_label']} (置信度: {result2['confidence']:.3f})")
    
    # 获取统计信息
    stats = model.get_statistics()
    print(f"\n统计信息:")
    print(f"总预测次数: {stats['total_predictions']}")
    print(f"平均置信度: {stats['average_confidence']:.3f}")
    print(f"校正率: {stats['correction_rate']:.3f}")
    print(f"愤怒/恐惧比例: {stats['anger_fear_ratio']:.3f}")

if __name__ == "__main__":
    print("M3D改进模型测试 v2.0")
    print("=" * 60)
    
    try:
        # 基础对比测试
        results = test_models()
        
        # 参数调整测试
        test_parameter_adjustment()
        
        print("\n测试完成!")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc() 