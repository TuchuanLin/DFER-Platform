"""
M3D模型推理包装器
提供统一的M3D模型推理接口
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from core_model.models.M3D import M3DFEL
from fer_platform.config import config

class M3DInferenceWrapper:
    """M3D模型推理包装器"""
    
    # 表情类别映射
    EMOTION_LABELS = {
        0: "愤怒",
        1: "厌恶", 
        2: "恐惧",
        3: "开心",
        4: "中性",
        5: "悲伤",
        6: "惊讶"
    }
    
    def __init__(self, model_path: str = None, device: str = "auto"):
        """
        初始化M3D推理包装器
        
        Args:
            model_path: 预训练模型路径，如果为None则使用配置文件中的路径
            device: 设备类型 ('auto', 'cpu', 'cuda')
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 使用配置文件中的模型路径（如果没有明确指定）
        if model_path is None:
            model_path = config.get_model_path()
        
        # 设置设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.logger.info(f"使用设备: {self.device}")
        
        # 初始化模型参数
        self.model_args = self._get_default_args()
        
        try:
            # 初始化模型
            self.model = M3DFEL(self.model_args)
            self.model.to(self.device)
            self.model.eval()
            
            # 加载预训练权重
            if model_path and Path(model_path).exists():
                self.load_model(model_path)
                self.logger.info(f"✅ 成功加载预训练模型: {model_path}")
            else:
                self._handle_no_pretrained_model(model_path)
                
        except Exception as e:
            self.logger.error(f"模型初始化失败: {e}")
            raise
    
    def _handle_no_pretrained_model(self, model_path: str = None):
        """处理没有预训练模型的情况"""
        if model_path:
            self.logger.warning(f"❌ 指定的模型文件不存在: {model_path}")
        
        model_info = config.get_model_info()
        
        self.logger.info("="*60)
        self.logger.info("🔍 预训练模型状态检查")
        self.logger.info("="*60)
        self.logger.info(f"📁 模型目录: {model_info['model_dir']}")
        self.logger.info(f"🔍 支持的模型文件: {', '.join(model_info['supported_files'])}")
        self.logger.info("💡 如需添加预训练模型，请将模型文件放在上述目录中")
        self.logger.info("⚠️  当前使用随机初始化权重，识别精度较低")
        self.logger.info("="*60)
        
        print("📝 提示: 当前使用随机初始化权重运行，如需更好的识别效果，请添加预训练模型")
    
    def _get_default_args(self):
        """获取默认模型参数"""
        class Args:
            def __init__(self):
                self.num_classes = config.MODEL_CONFIG['num_classes']
                self.num_frames = config.MODEL_CONFIG['num_frames']
                self.instance_length = 4
                self.crop_size = config.MODEL_CONFIG['crop_size']
                self.gpu_ids = [0] if torch.cuda.is_available() else []
        
        return Args()
    
    def load_model(self, model_path: str):
        """加载预训练模型"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 处理不同的checkpoint格式
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                
            self.model.load_state_dict(state_dict, strict=False)
            self.logger.info(f"成功加载模型: {model_path}")
            
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            raise
    
    def preprocess_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        预处理视频帧
        
        Args:
            frames: 原始视频帧列表
            
        Returns:
            preprocessed_tensor: 预处理后的张量 [B, T, C, H, W]
        """
        try:
            # 确保帧数为16
            if len(frames) != 16:
                frames = self._sample_frames(frames, 16)
            
            processed_frames = []
            for frame in frames:
                # 确保frame是numpy数组
                if not isinstance(frame, np.ndarray):
                    continue
                    
                # 调整大小到112x112
                frame = cv2.resize(frame, (112, 112))
                
                # BGR转RGB
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                elif len(frame.shape) == 2:
                    # 灰度图转RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                
                # 归一化到[0,1]
                frame = frame.astype(np.float32) / 255.0
                
                # 转换为CHW格式
                frame = np.transpose(frame, (2, 0, 1))
                processed_frames.append(frame)
            
            # 确保有16帧
            while len(processed_frames) < 16:
                if processed_frames:
                    processed_frames.append(processed_frames[-1])
                else:
                    # 创建空白帧
                    blank_frame = np.zeros((3, 112, 112), dtype=np.float32)
                    processed_frames.append(blank_frame)
            
            # 堆叠为tensor: [T, C, H, W]
            tensor = torch.tensor(np.stack(processed_frames), dtype=torch.float32)
            
            # 添加batch维度并调整为正确格式: [B, T, C, H, W] -> [1, 16, 3, 112, 112]
            tensor = tensor.unsqueeze(0)
            
            # 确保tensor是连续的，避免stride问题
            tensor = tensor.contiguous()
            
            return tensor.to(self.device)
            
        except Exception as e:
            self.logger.error(f"预处理帧时出错: {e}")
            # 返回默认tensor
            return torch.zeros(1, 16, 3, 112, 112, dtype=torch.float32, device=self.device)
    
    def _sample_frames(self, frames: List[np.ndarray], target_num: int) -> List[np.ndarray]:
        """均匀采样视频帧"""
        if len(frames) <= target_num:
            # 如果帧数不足，重复最后一帧
            return frames + [frames[-1]] * (target_num - len(frames))
        
        # 均匀采样
        indices = np.linspace(0, len(frames) - 1, target_num, dtype=int)
        return [frames[i] for i in indices]
    
    @torch.no_grad()
    def predict(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        预测表情
        
        Args:
            frames: 视频帧列表
            
        Returns:
            prediction_result: 预测结果字典
        """
        try:
            # 预处理
            input_tensor = self.preprocess_frames(frames)
            self.logger.debug(f"输入tensor形状: {input_tensor.shape}")
            
            # 推理
            output, features = self.model(input_tensor)
            self.logger.debug(f"模型输出形状: {output.shape}")
            
            # 确保输出是2维的 [batch, num_classes]
            if output.dim() > 2:
                # 如果输出维度大于2，进行reshape
                output = output.view(output.size(0), -1)
                if output.size(1) != self.model_args.num_classes:
                    # 如果不是预期的类别数，取最后几维
                    output = output[:, -self.model_args.num_classes:]
            elif output.dim() == 1:
                output = output.unsqueeze(0)
            
            # 确保输出维度正确
            if output.size(1) != self.model_args.num_classes:
                self.logger.warning(f"输出维度异常: {output.shape}, 期望: {self.model_args.num_classes}")
                # 创建默认输出
                output = torch.zeros(1, self.model_args.num_classes, device=self.device)
                output[0, 2] = 1.0  # 默认为中性表情
            
            # 计算概率
            probabilities = F.softmax(output, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
            
            # 构建结果（注意：去除numpy数组，避免JSON序列化问题）
            result = {
                'predicted_class': predicted_class,
                'predicted_label': self.EMOTION_LABELS[predicted_class],
                'confidence': confidence,
                'probabilities': {
                    self.EMOTION_LABELS[i]: prob.item() 
                    for i, prob in enumerate(probabilities[0])
                },
                'model_info': {
                    'has_pretrained': config.has_pretrained_model(),
                    'device': str(self.device)
                }
                # 移除features字段避免序列化问题
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"预测过程出错: {e}")
            # 返回默认结果
            return {
                'predicted_class': 4,
                'predicted_label': self.EMOTION_LABELS[4],
                'confidence': 0.5,
                'probabilities': {label: 1.0/7 for label in self.EMOTION_LABELS.values()},
                'model_info': {
                    'has_pretrained': config.has_pretrained_model(),
                    'device': str(self.device)
                },
                'error': str(e)
            }
    
    def predict_batch(self, batch_frames: List[List[np.ndarray]]) -> List[Dict[str, Any]]:
        """
        批量预测
        
        Args:
            batch_frames: 批量视频帧列表
            
        Returns:
            batch_results: 批量预测结果
        """
        return [self.predict(frames) for frames in batch_frames]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            model_info: 模型信息字典
        """
        model_config_info = config.get_model_info()
        
        return {
            'model_name': 'M3D-FER',
            'num_classes': self.model_args.num_classes,
            'emotion_labels': self.EMOTION_LABELS,
            'device': str(self.device),
            'input_shape': [1, 16, 3, 112, 112],
            'pretrained_info': model_config_info
        } 