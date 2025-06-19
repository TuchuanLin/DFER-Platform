"""
视频处理器
支持多种视频格式的读取、处理和帧提取
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Generator
import logging
from pathlib import Path
import tempfile
import os


class VideoProcessor:
    """视频处理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 支持的视频格式
        self.supported_formats = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    
    def validate_video_format(self, file_path: str) -> bool:
        """验证视频格式是否支持"""
        return Path(file_path).suffix.lower() in self.supported_formats
    
    def extract_frames_from_file(self, video_path: str, max_frames: int = None) -> List[np.ndarray]:
        """
        从视频文件提取帧
        
        Args:
            video_path: 视频文件路径
            max_frames: 最大帧数限制
            
        Returns:
            frames: 提取的帧列表
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        if not self.validate_video_format(video_path):
            raise ValueError(f"不支持的视频格式: {Path(video_path).suffix}")
        
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        try:
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件: {video_path}")
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frames.append(frame)
                frame_count += 1
                
                if max_frames and frame_count >= max_frames:
                    break
                    
            self.logger.info(f"从视频提取了 {len(frames)} 帧")
            
        finally:
            cap.release()
        
        return frames
    
    def extract_frames_from_bytes(self, video_bytes: bytes, max_frames: int = None) -> List[np.ndarray]:
        """
        从视频字节数据提取帧
        
        Args:
            video_bytes: 视频字节数据
            max_frames: 最大帧数限制
            
        Returns:
            frames: 提取的帧列表
        """
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_file.write(video_bytes)
            temp_path = temp_file.name
        
        try:
            return self.extract_frames_from_file(temp_path, max_frames)
        finally:
            # 清理临时文件
            try:
                os.unlink(temp_path)
            except OSError:
                pass
    
    def get_video_info(self, video_path: str) -> dict:
        """
        获取视频信息
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            video_info: 视频信息字典
        """
        cap = cv2.VideoCapture(video_path)
        
        try:
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件: {video_path}")
            
            info = {
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'duration': 0,
                'format': Path(video_path).suffix.lower()
            }
            
            if info['fps'] > 0:
                info['duration'] = info['frame_count'] / info['fps']
            
            return info
            
        finally:
            cap.release()
    
    def sample_frames_uniformly(self, frames: List[np.ndarray], target_count: int) -> List[np.ndarray]:
        """
        均匀采样视频帧
        
        Args:
            frames: 原始帧列表
            target_count: 目标帧数
            
        Returns:
            sampled_frames: 采样后的帧列表
        """
        if len(frames) <= target_count:
            # 帧数不足时，重复最后一帧
            return frames + [frames[-1]] * (target_count - len(frames))
        
        # 均匀采样
        indices = np.linspace(0, len(frames) - 1, target_count, dtype=int)
        return [frames[i] for i in indices]
    
    def resize_frame(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        调整帧大小
        
        Args:
            frame: 原始帧
            target_size: 目标大小 (width, height)
            
        Returns:
            resized_frame: 调整后的帧
        """
        return cv2.resize(frame, target_size)
    
    def crop_center(self, frame: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
        """
        中心裁剪
        
        Args:
            frame: 原始帧
            crop_size: 裁剪大小 (width, height)
            
        Returns:
            cropped_frame: 裁剪后的帧
        """
        h, w = frame.shape[:2]
        crop_w, crop_h = crop_size
        
        start_x = max(0, (w - crop_w) // 2)
        start_y = max(0, (h - crop_h) // 2)
        
        return frame[start_y:start_y + crop_h, start_x:start_x + crop_w]
    
    def normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        归一化帧到[0,1]范围
        
        Args:
            frame: 原始帧
            
        Returns:
            normalized_frame: 归一化后的帧
        """
        return frame.astype(np.float32) / 255.0


class CameraProcessor:
    """摄像头处理器"""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def start_camera(self) -> bool:
        """启动摄像头"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                raise RuntimeError(f"无法打开摄像头 {self.camera_id}")
            
            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.logger.info(f"摄像头 {self.camera_id} 启动成功")
            return True
            
        except Exception as e:
            self.logger.error(f"启动摄像头失败: {e}")
            return False
    
    def stop_camera(self):
        """停止摄像头"""
        if self.cap:
            self.cap.release()
            self.cap = None
            self.logger.info("摄像头已停止")
    
    def read_frame(self) -> Optional[np.ndarray]:
        """读取一帧"""
        if not self.cap or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def read_frames_sequence(self, frame_count: int) -> List[np.ndarray]:
        """读取连续帧序列"""
        frames = []
        for _ in range(frame_count):
            frame = self.read_frame()
            if frame is not None:
                frames.append(frame)
        
        return frames
    
    def get_camera_info(self) -> dict:
        """获取摄像头信息"""
        if not self.cap or not self.cap.isOpened():
            return {}
        
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'camera_id': self.camera_id
        } 