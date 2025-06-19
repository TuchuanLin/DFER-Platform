"""
任务管理器
管理视频处理任务的生命周期和状态
"""

from typing import Dict, Any, List, Optional
import threading
import time
from datetime import datetime, timedelta
import logging


class TaskManager:
    """任务管理器"""
    
    def __init__(self, max_tasks: int = 1000, cleanup_interval: int = 3600):
        """
        初始化任务管理器
        
        Args:
            max_tasks: 最大任务数量
            cleanup_interval: 清理间隔（秒）
        """
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.max_tasks = max_tasks
        self.cleanup_interval = cleanup_interval
        self.lock = threading.RLock()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 启动清理线程
        self._start_cleanup_thread()
    
    def create_task(self, task_id: str, task_info: Dict[str, Any]) -> bool:
        """
        创建新任务
        
        Args:
            task_id: 任务ID
            task_info: 任务信息
            
        Returns:
            success: 是否创建成功
        """
        with self.lock:
            if len(self.tasks) >= self.max_tasks:
                self._cleanup_old_tasks()
                
                if len(self.tasks) >= self.max_tasks:
                    self.logger.warning("任务数量已达上限，无法创建新任务")
                    return False
            
            self.tasks[task_id] = {
                **task_info,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            self.logger.info(f"创建任务: {task_id}")
            return True
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务信息
        
        Args:
            task_id: 任务ID
            
        Returns:
            task_info: 任务信息，不存在时返回None
        """
        with self.lock:
            return self.tasks.get(task_id)
    
    def update_task_status(self, task_id: str, status: str) -> bool:
        """
        更新任务状态
        
        Args:
            task_id: 任务ID
            status: 新状态
            
        Returns:
            success: 是否更新成功
        """
        with self.lock:
            if task_id not in self.tasks:
                return False
            
            self.tasks[task_id]['status'] = status
            self.tasks[task_id]['updated_at'] = datetime.now().isoformat()
            
            self.logger.info(f"更新任务状态: {task_id} -> {status}")
            return True
    
    def update_task_result(self, task_id: str, result: Dict[str, Any]) -> bool:
        """
        更新任务结果
        
        Args:
            task_id: 任务ID
            result: 任务结果
            
        Returns:
            success: 是否更新成功
        """
        with self.lock:
            if task_id not in self.tasks:
                return False
            
            self.tasks[task_id]['result'] = result
            self.tasks[task_id]['updated_at'] = datetime.now().isoformat()
            
            self.logger.info(f"更新任务结果: {task_id}")
            return True
    
    def delete_task(self, task_id: str) -> bool:
        """
        删除任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            success: 是否删除成功
        """
        with self.lock:
            if task_id not in self.tasks:
                return False
            
            del self.tasks[task_id]
            self.logger.info(f"删除任务: {task_id}")
            return True
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """
        获取所有任务列表
        
        Returns:
            tasks: 所有任务列表
        """
        with self.lock:
            return list(self.tasks.values())
    
    def get_tasks_by_status(self, status: str) -> List[Dict[str, Any]]:
        """
        根据状态获取任务列表
        
        Args:
            status: 任务状态
            
        Returns:
            tasks: 指定状态的任务列表
        """
        with self.lock:
            return [task for task in self.tasks.values() if task.get('status') == status]
    
    def get_task_count(self) -> Dict[str, int]:
        """
        获取任务统计信息
        
        Returns:
            statistics: 任务统计信息
        """
        with self.lock:
            total = len(self.tasks)
            processing = len([t for t in self.tasks.values() if t.get('status') == 'processing'])
            completed = len([t for t in self.tasks.values() if t.get('status') == 'completed'])
            failed = len([t for t in self.tasks.values() if t.get('status') == 'failed'])
            
            return {
                'total': total,
                'processing': processing,
                'completed': completed,
                'failed': failed,
                'pending': total - processing - completed - failed
            }
    
    def _cleanup_old_tasks(self, max_age_hours: int = 24):
        """
        清理旧任务
        
        Args:
            max_age_hours: 最大保留时间（小时）
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        tasks_to_remove = []
        for task_id, task_info in self.tasks.items():
            try:
                created_at = datetime.fromisoformat(task_info['created_at'])
                if created_at < cutoff_time:
                    tasks_to_remove.append(task_id)
            except (KeyError, ValueError):
                # 如果时间格式有问题，也删除
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self.tasks[task_id]
        
        if tasks_to_remove:
            self.logger.info(f"清理了 {len(tasks_to_remove)} 个旧任务")
    
    def _start_cleanup_thread(self):
        """启动清理线程"""
        def cleanup_worker():
            while True:
                time.sleep(self.cleanup_interval)
                try:
                    with self.lock:
                        self._cleanup_old_tasks()
                except Exception as e:
                    self.logger.error(f"清理任务时出错: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        self.logger.info("任务清理线程已启动")
    
    def clear_all_tasks(self):
        """清空所有任务（慎用）"""
        with self.lock:
            count = len(self.tasks)
            self.tasks.clear()
            self.logger.warning(f"清空了所有任务，共 {count} 个")
    
    def get_task_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取任务历史（按创建时间降序）
        
        Args:
            limit: 返回数量限制
            
        Returns:
            history: 任务历史列表
        """
        with self.lock:
            sorted_tasks = sorted(
                self.tasks.values(),
                key=lambda x: x.get('created_at', ''),
                reverse=True
            )
            return sorted_tasks[:limit] 