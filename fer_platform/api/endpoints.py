# -*- coding: utf-8 -*-
"""
FastAPI endpoints for FER Platform
"""

import os
import uuid
import asyncio
import logging
import json
import base64
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from fer_platform.models.m3d_wrapper import M3DInferenceWrapper
from fer_platform.core.video_processor import VideoProcessor, CameraProcessor
from fer_platform.core.task_manager import TaskManager

# 初始化组件
video_processor = VideoProcessor()

# 选择模型包装器 - 使用原始预训练权重输出
from fer_platform.models.improved_m3d_wrapper import MICACLInferenceWrapper
model_wrapper = MICACLInferenceWrapper(
    enable_bias_correction=False,     # 禁用偏差校正，使用原始权重
    enable_confidence_filtering=False, # 禁用置信度过滤
    confidence_threshold=0.0          # 不设置阈值
)

task_manager = TaskManager()

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload", summary="上传视频进行表情识别")
async def upload_video(file: UploadFile = File(...)):
    """
    上传视频文件进行表情识别
    
    Args:
        file: 上传的视频文件
        
    Returns:
        task_info: 任务信息和结果
    """
    try:
        # 验证文件格式
        if not video_processor.validate_video_format(file.filename):
            raise HTTPException(
                status_code=400, 
                detail=f"不支持的视频格式。支持的格式: {', '.join(video_processor.supported_formats)}"
            )
        
        # 读取文件内容
        content = await file.read()
        
        # 生成任务ID
        task_id = str(uuid.uuid4())
        
        # 创建任务
        task_info = {
            'task_id': task_id,
            'filename': file.filename,
            'file_size': len(content),
            'status': 'processing',
            'created_at': datetime.now().isoformat(),
            'result': None
        }
        
        # 存储任务信息
        task_manager.create_task(task_id, task_info)
        
        # 异步处理视频
        asyncio.create_task(process_uploaded_video(task_id, content, file.filename))
        
        return JSONResponse({
            'task_id': task_id,
            'status': 'accepted',
            'message': '视频上传成功，正在处理中...',
            'filename': file.filename
        })
        
    except Exception as e:
        logger.error(f"视频上传处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


async def process_uploaded_video(task_id: str, video_content: bytes, filename: str):
    """异步处理上传的视频"""
    try:
        # 更新任务状态
        task_manager.update_task_status(task_id, 'processing')
        
        # 提取视频帧
        frames = video_processor.extract_frames_from_bytes(video_content)
        
        if not frames:
            raise ValueError("无法从视频中提取帧")
        
        # 进行表情识别
        result = model_wrapper.predict(frames)
        
        # 计算额外统计信息
        additional_info = {
            'total_frames': len(frames),
            'processed_frames': 16,  # M3D模型使用16帧
            'processing_time': datetime.now().isoformat(),
            'video_info': {
                'filename': filename,
                'frame_count': len(frames)
            }
        }
        
        result.update(additional_info)
        
        # 更新任务结果
        task_manager.update_task_result(task_id, result)
        task_manager.update_task_status(task_id, 'completed')
        
    except Exception as e:
        logger.error(f"视频处理失败 {task_id}: {e}")
        error_result = {
            'error': str(e),
            'error_type': type(e).__name__
        }
        task_manager.update_task_result(task_id, error_result)
        task_manager.update_task_status(task_id, 'failed')


@router.get("/result/{task_id}", summary="获取识别结果")
async def get_result(task_id: str):
    """
    获取任务识别结果
    
    Args:
        task_id: 任务ID
        
    Returns:
        task_result: 任务结果
    """
    try:
        task_info = task_manager.get_task(task_id)
        
        if not task_info:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        return JSONResponse(task_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取结果失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取结果失败: {str(e)}")


@router.get("/tasks", summary="获取所有任务列表")
async def get_all_tasks():
    """获取所有任务的列表"""
    try:
        tasks = task_manager.get_all_tasks()
        return JSONResponse({
            'total': len(tasks),
            'tasks': tasks
        })
    except Exception as e:
        logger.error(f"获取任务列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取任务列表失败: {str(e)}")


@router.delete("/task/{task_id}", summary="删除任务")
async def delete_task(task_id: str):
    """删除指定任务"""
    try:
        success = task_manager.delete_task(task_id)
        if not success:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        return JSONResponse({
            'message': '任务删除成功',
            'task_id': task_id
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除任务失败: {str(e)}")


@router.get("/model/info", summary="获取模型信息")
async def get_model_info():
    """获取模型信息"""
    try:
        if hasattr(model_wrapper, 'get_model_info'):
            model_info = model_wrapper.get_model_info()
        else:
            model_info = {"model_name": "MICACL-FER", "num_classes": 7}
        
        # 添加改进模型的特定信息
        if hasattr(model_wrapper, 'BIAS_CORRECTION_WEIGHTS'):
            model_info['improvement_features'] = {
                'type': 'improved_model',
                'bias_correction_enabled': getattr(model_wrapper, 'enable_bias_correction', False),
                'confidence_filtering_enabled': getattr(model_wrapper, 'enable_confidence_filtering', False),
                'bias_correction_weights': model_wrapper.BIAS_CORRECTION_WEIGHTS,
                'temperature': getattr(model_wrapper, 'TEMPERATURE', 1.0),
                'confidence_threshold': getattr(model_wrapper, 'confidence_threshold', 0.5)
            }
        else:
            model_info['improvement_features'] = {
                'type': 'original_model',
                'message': '当前使用原版模型，未启用改进功能'
            }
        
        return JSONResponse(model_info)
    except Exception as e:
        logger.error(f"获取模型信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取模型信息失败: {str(e)}")

@router.post("/model/configure", summary="配置模型参数")
async def configure_model_parameters(
    bias_correction_weights: Optional[Dict[int, float]] = None,
    temperature: Optional[float] = None,
    confidence_threshold: Optional[float] = None,
    enable_bias_correction: Optional[bool] = None,
    enable_confidence_filtering: Optional[bool] = None
):
    """配置改进模型的参数"""
    try:
        if not hasattr(model_wrapper, 'configure_parameters'):
            return JSONResponse({
                "message": "当前模型不支持参数配置",
                "type": "original_model"
            })
        
        result = model_wrapper.configure_parameters(
            bias_correction_weights=bias_correction_weights,
            temperature=temperature,
            confidence_threshold=confidence_threshold,
            enable_bias_correction=enable_bias_correction,
            enable_confidence_filtering=enable_confidence_filtering
        )
        
        return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"配置模型参数失败: {e}")
        raise HTTPException(status_code=500, detail=f"配置模型参数失败: {str(e)}")

@router.post("/model/reset-statistics", summary="重置模型统计")
async def reset_model_statistics():
    """重置模型预测统计信息"""
    try:
        if not hasattr(model_wrapper, 'reset_statistics'):
            return JSONResponse({
                "message": "当前模型不支持统计重置",
                "type": "original_model"
            })
        
        model_wrapper.reset_statistics()
        
        return JSONResponse({
            "status": "success",
            "message": "统计信息已重置"
        })
        
    except Exception as e:
        logger.error(f"重置统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"重置统计信息失败: {str(e)}")

@router.get("/model/statistics", summary="获取模型预测统计")
async def get_model_statistics():
    """获取模型预测统计和改进效果"""
    try:
        if not hasattr(model_wrapper, 'emotion_stats'):
            return JSONResponse({
                "message": "当前模型不支持统计功能",
                "type": "original_model"
            })
        
        total_predictions = len(getattr(model_wrapper, 'prediction_history', []))
        emotion_stats = getattr(model_wrapper, 'emotion_stats', {})
        
        # 计算愤怒和恐惧的比例
        anger_fear_count = emotion_stats.get('愤怒', 0) + emotion_stats.get('恐惧', 0)
        anger_fear_ratio = anger_fear_count / total_predictions if total_predictions > 0 else 0
        
        return JSONResponse({
            "total_predictions": total_predictions,
            "emotion_distribution": emotion_stats,
            "anger_fear_ratio": anger_fear_ratio,
            "anger_fear_count": anger_fear_count,
            "bias_correction_status": "启用" if getattr(model_wrapper, 'enable_bias_correction', False) else "禁用",
            "confidence_filtering_status": "启用" if getattr(model_wrapper, 'enable_confidence_filtering', False) else "禁用",
            "current_threshold": getattr(model_wrapper, 'confidence_threshold', 0.5),
            "recommendation": {
                "status": "良好" if anger_fear_ratio < 0.4 else "建议调整",
                "message": f"愤怒和恐惧预测比例: {anger_fear_ratio:.2%}",
                "suggestion": "表现良好" if anger_fear_ratio < 0.4 else "建议启用或调整偏差校正参数"
            }
        })
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


@router.get("/model/detailed-statistics", summary="获取详细模型统计")
async def get_detailed_model_statistics():
    """获取详细的模型预测统计信息"""
    try:
        if not hasattr(model_wrapper, 'get_statistics'):
            return JSONResponse({
                "message": "当前模型不支持详细统计",
                "type": "original_model"
            })
        
        stats = model_wrapper.get_statistics()
        return JSONResponse(stats)
        
    except Exception as e:
        logger.error(f"获取详细统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取详细统计信息失败: {str(e)}")


# WebSocket连接管理器
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()


@router.websocket("/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """实时表情识别WebSocket接口"""
    await manager.connect(websocket)
    
    try:
        # 发送连接成功消息
        await manager.send_personal_message(
            json.dumps({
                'type': 'connection',
                'status': 'connected',
                'message': '实时识别连接已建立'
            }), 
            websocket
        )
        
        while True:
            try:
                # 接收客户端发送的图像数据（二进制数据）
                data = await websocket.receive_bytes()
                
                # 将字节数据转换为图像
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None and frame.size > 0:
                    # 调整图像大小为模型需要的尺寸
                    frame = cv2.resize(frame, (112, 112))
                    
                    # 复制16帧（M3D模型需要16帧输入）
                    frames = [frame.copy() for _ in range(16)]
                    
                    # 进行预测
                    result = model_wrapper.predict(frames)
                    
                    # 发送预测结果 - 确保所有数值都可序列化
                    response_data = {
                        'type': 'prediction_result',
                        'emotion': str(result['predicted_label']),
                        'confidence': float(result['confidence']),
                        'probabilities': {str(k): float(v) for k, v in result['probabilities'].items()},
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # 如果预测结果包含错误信息，也发送给客户端
                    if 'error' in result:
                        response_data['warning'] = str(result['error'])
                    
                    try:
                        await manager.send_personal_message(
                            json.dumps(response_data, ensure_ascii=False),
                            websocket
                        )
                    except Exception as json_error:
                        logger.error(f"JSON序列化错误: {json_error}")
                        # 发送简化的错误消息
                        simple_response = {
                            'type': 'error',
                            'message': 'JSON序列化失败'
                        }
                        await manager.send_personal_message(
                            json.dumps(simple_response),
                            websocket
                        )
                else:
                    # 发送错误消息
                    await manager.send_personal_message(
                        json.dumps({
                            'type': 'error',
                            'message': '无法解析图像数据'
                        }),
                        websocket
                    )
                    
            except Exception as e:
                logger.error(f"帧处理错误: {e}")
                try:
                    error_message = {
                        'type': 'error',
                        'message': f'处理失败: {str(e)}'
                    }
                    await manager.send_personal_message(
                        json.dumps(error_message, ensure_ascii=False),
                        websocket
                    )
                except Exception as send_error:
                    logger.error(f"发送错误消息失败: {send_error}")
                    # 如果发送失败，说明连接已断开
                    break
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket连接断开")
    
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
        try:
            manager.disconnect(websocket)
        except:
            pass


@router.get("/health", summary="健康检查")
async def health_check():
    """健康检查接口"""
    return JSONResponse({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }) 