from typing import List
import numpy as np
import cv2
import torch

class M3DWrapper:
    def __init__(self, device):
        self.device = device
        self.logger = None  # Assuming a logger is set up

    def preprocess_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        预处理视频帧 - 完全重写以避免tensor错误
        
        Args:
            frames: 原始视频帧列表
            
        Returns:
            preprocessed_tensor: 预处理后的张量 [B, T, C, H, W]
        """
        try:
            # 确保至少有一帧
            if not frames or len(frames) == 0:
                # 创建默认黑色帧
                frames = [np.zeros((112, 112, 3), dtype=np.uint8) for _ in range(16)]
            
            # 处理每一帧
            processed_frames = []
            for i in range(16):  # 固定处理16帧
                if i < len(frames):
                    frame = frames[i]
                else:
                    # 重复最后一帧
                    frame = frames[-1] if frames else np.zeros((112, 112, 3), dtype=np.uint8)
                
                # 确保frame是numpy数组
                if not isinstance(frame, np.ndarray):
                    frame = np.zeros((112, 112, 3), dtype=np.uint8)
                
                # 调整大小和格式
                if len(frame.shape) == 2:  # 灰度图
                    frame = np.stack([frame, frame, frame], axis=-1)
                elif len(frame.shape) == 3 and frame.shape[2] == 1:  # 单通道
                    frame = np.repeat(frame, 3, axis=2)
                elif len(frame.shape) == 3 and frame.shape[2] == 4:  # RGBA
                    frame = frame[:, :, :3]
                
                # 调整大小
                if frame.shape[:2] != (112, 112):
                    frame = cv2.resize(frame, (112, 112))
                
                # 确保是RGB格式
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    # 假设输入是BGR，转换为RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 归一化
                frame = frame.astype(np.float32) / 255.0
                
                # 转换为CHW格式
                frame = np.transpose(frame, (2, 0, 1))  # (H, W, C) -> (C, H, W)
                
                processed_frames.append(frame)
            
            # 转换为numpy数组然后到tensor
            frames_array = np.stack(processed_frames, axis=0)  # (T, C, H, W)
            tensor = torch.from_numpy(frames_array).float()
            
            # 添加batch维度
            tensor = tensor.unsqueeze(0)  # (1, T, C, H, W)
            
            # 确保在正确设备上
            tensor = tensor.to(self.device)
            
            return tensor
            
        except Exception as e:
            self.logger.error(f"预处理失败: {e}")
            # 返回安全的默认tensor
            default_tensor = torch.zeros(1, 16, 3, 112, 112, dtype=torch.float32, device=self.device)
            return default_tensor 