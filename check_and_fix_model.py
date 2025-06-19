#!/usr/bin/env python3
"""
检查和修复模型问题的专用工具
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
import logging

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from fer_platform.models.improved_m3d_wrapper import ImprovedM3DInferenceWrapper
from fer_platform.config import config
from core_model.models.M3D import M3DFEL

def check_model_weights():
    """检查模型权重是否正常加载"""
    print("🔍 检查模型权重状态...")
    print("=" * 50)
    
    model_path = config.get_model_path()
    print(f"模型文件路径: {model_path}")
    
    if not model_path or not Path(model_path).exists():
        print("❌ 模型文件不存在")
        return False
    
    try:
        # 检查模型文件
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"✅ 模型文件加载成功")
        
        # 分析checkpoint结构
        if isinstance(checkpoint, dict):
            print(f"Checkpoint 键: {list(checkpoint.keys())}")
            
            # 查找state_dict
            state_dict = None
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("使用 'model_state_dict'")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("使用 'state_dict'")
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
                print("使用 'model'")
            else:
                state_dict = checkpoint
                print("直接使用 checkpoint 作为 state_dict")
                
            if state_dict:
                print(f"State dict 中的层数: {len(state_dict)}")
                print("前10个参数层:")
                for i, (name, param) in enumerate(list(state_dict.items())[:10]):
                    print(f"  {i+1}. {name}: {param.shape}")
                    
                # 检查参数是否为零或异常
                zero_params = 0
                total_params = 0
                for name, param in state_dict.items():
                    total_params += 1
                    if torch.all(param == 0):
                        zero_params += 1
                        
                print(f"参数统计: {total_params} 总参数, {zero_params} 零参数")
                if zero_params > total_params * 0.1:  # 如果超过10%是零参数
                    print("⚠️  警告: 发现异常多的零参数")
                    
        return True
        
    except Exception as e:
        print(f"❌ 检查模型权重时出错: {e}")
        return False

def test_model_predictions():
    """测试模型的基本预测能力"""
    print("\n🧪 测试模型预测能力...")
    print("=" * 50)
    
    try:
        # 创建不同强度的偏差校正版本
        configs = [
            {'name': '无校正', 'enable_bias_correction': False},
            {'name': '激进校正', 'enable_bias_correction': True}
        ]
        
        # 创建测试数据
        test_frame = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        test_frames = [test_frame for _ in range(16)]
        
        for config_test in configs:
            print(f"\n📊 配置: {config_test['name']}")
            
            wrapper = ImprovedM3DInferenceWrapper(
                enable_bias_correction=config_test['enable_bias_correction'],
                enable_confidence_filtering=True
            )
            
            # 进行多次预测，检查一致性
            predictions = []
            for i in range(5):
                result = wrapper.predict(test_frames)
                predictions.append(result['predicted_label'])
                
            # 分析预测一致性
            unique_predictions = set(predictions)
            print(f"  5次预测结果: {predictions}")
            print(f"  预测一致性: {len(unique_predictions) == 1}")
            
            # 显示最后一次的详细结果
            result = wrapper.predict(test_frames)
            probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
            print(f"  最终预测: {result['predicted_label']} (置信度: {result['confidence']:.4f})")
            print(f"  概率分布:")
            for emotion, prob in probs:
                print(f"    {emotion}: {prob:.4f}")
                
    except Exception as e:
        print(f"❌ 测试预测时出错: {e}")
        import traceback
        traceback.print_exc()

def apply_extreme_bias_correction():
    """应用极端的偏差校正来解决中性过度预测"""
    print("\n🎯 应用极端偏差校正...")
    print("=" * 50)
    
    try:
        wrapper = ImprovedM3DInferenceWrapper()
        
        # 设置极端的偏差校正权重
        extreme_weights = {
            0: 3.0,   # 愤怒 - 极大提高
            1: 2.5,   # 厌恶 - 极大提高
            2: 3.5,   # 恐惧 - 极大提高
            3: 4.0,   # 开心 - 最大提高
            4: 0.1,   # 中性 - 极大降低
            5: 3.0,   # 悲伤 - 极大提高
            6: 3.5    # 惊讶 - 极大提高
        }
        
        wrapper.configure_parameters(
            bias_correction_weights=extreme_weights,
            temperature=0.5,  # 更低的温度
            confidence_threshold=0.1
        )
        
        # 创建测试数据
        test_frame = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        test_frames = [test_frame for _ in range(16)]
        
        print("使用极端偏差校正进行预测...")
        result = wrapper.predict(test_frames)
        
        print(f"预测结果: {result['predicted_label']}")
        print(f"置信度: {result['confidence']:.4f}")
        
        probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
        print("概率分布:")
        for emotion, prob in probs:
            symbol = "🔥" if emotion != "中性" else "❄️"
            print(f"  {symbol} {emotion}: {prob:.4f}")
            
        if result['predicted_label'] != "中性":
            print("✅ 成功避免中性过度预测！")
        else:
            print("⚠️  仍然预测为中性，可能需要更极端的调整")
            
    except Exception as e:
        print(f"❌ 应用极端偏差校正时出错: {e}")
        import traceback
        traceback.print_exc()

def diagnose_model_architecture():
    """诊断模型架构问题"""
    print("\n🏗️  诊断模型架构...")
    print("=" * 50)
    
    try:
        # 直接创建和检查M3D模型
        class Args:
            def __init__(self):
                self.num_classes = 7
                self.num_frames = 16
                self.instance_length = 4
                self.gpu_ids = []
        
        args = Args()
        model = M3DFEL(args)
        
        print(f"模型创建成功")
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
        print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
        # 测试前向传播
        test_input = torch.randn(1, 16, 3, 112, 112)
        print(f"测试输入形状: {test_input.shape}")
        
        model.eval()
        with torch.no_grad():
            output, features = model(test_input)
            print(f"模型输出形状: {output.shape}")
            print(f"输出数值范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
            
            # 检查输出是否异常
            if torch.isnan(output).any():
                print("⚠️  警告: 输出包含 NaN")
            if torch.isinf(output).any():
                print("⚠️  警告: 输出包含 Inf")
                
            # 查看原始概率分布
            probs = torch.softmax(output, dim=-1)
            print("原始概率分布 (softmax):")
            emotion_labels = ["愤怒", "厌恶", "恐惧", "开心", "中性", "悲伤", "惊讶"]
            for i, (emotion, prob) in enumerate(zip(emotion_labels, probs[0])):
                print(f"  {emotion}: {prob.item():.4f}")
                
        return True
        
    except Exception as e:
        print(f"❌ 诊断模型架构时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🔧 模型诊断和修复工具")
    print("=" * 60)
    
    # 1. 检查模型权重
    weights_ok = check_model_weights()
    
    # 2. 诊断模型架构
    arch_ok = diagnose_model_architecture()
    
    # 3. 测试预测能力
    test_model_predictions()
    
    # 4. 应用极端偏差校正
    apply_extreme_bias_correction()
    
    print("\n" + "=" * 60)
    print("🏁 诊断完成")
    print("=" * 60)
    
    if weights_ok and arch_ok:
        print("✅ 模型文件和架构正常")
        print("💡 建议: 问题可能在于预训练模型本身偏向中性预测")
        print("💡 解决方案: 使用更激进的偏差校正或寻找更好的预训练模型")
    else:
        print("❌ 发现模型问题，需要检查模型文件或架构")

if __name__ == "__main__":
    main() 