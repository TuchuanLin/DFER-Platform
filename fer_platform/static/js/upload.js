// 视频上传功能的JavaScript代码

class VideoUploader {
    constructor() {
        this.initializeElements();
        this.bindEvents();
        this.currentTaskId = null;
        this.pollInterval = null;
        
        // 表情图标映射
        this.emotionIcons = {
            '快乐 (Happy)': '😊',
            '悲伤 (Sad)': '😢',
            '中性 (Neutral)': '😐',
            '愤怒 (Angry)': '😠',
            '惊讶 (Surprise)': '😲',
            '厌恶 (Disgust)': '🤢',
            '恐惧 (Fear)': '😨'
        };
    }

    initializeElements() {
        this.uploadArea = document.getElementById('uploadArea');
        this.videoFile = document.getElementById('videoFile');
        this.fileInfo = document.getElementById('fileInfo');
        this.fileDetails = document.getElementById('fileDetails');
        this.uploadProgress = document.getElementById('uploadProgress');
        this.progressBar = document.getElementById('progressBar');
        this.progressText = document.getElementById('progressText');
        this.uploadBtn = document.getElementById('uploadBtn');
        this.cancelBtn = document.getElementById('cancelBtn');
        this.processingStatus = document.getElementById('processingStatus');
        this.resultSection = document.getElementById('resultSection');
        this.errorSection = document.getElementById('errorSection');
        this.taskId = document.getElementById('taskId');
    }

    bindEvents() {
        // 点击上传区域
        this.uploadArea.addEventListener('click', () => {
            this.videoFile.click();
        });

        // 文件选择
        this.videoFile.addEventListener('change', (e) => {
            this.handleFileSelect(e.target.files[0]);
        });

        // 拖拽上传
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.classList.add('dragover');
        });

        this.uploadArea.addEventListener('dragleave', () => {
            this.uploadArea.classList.remove('dragover');
        });

        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelect(files[0]);
            }
        });

        // 上传按钮
        this.uploadBtn.addEventListener('click', () => {
            this.uploadVideo();
        });

        // 取消按钮
        this.cancelBtn.addEventListener('click', () => {
            this.cancelUpload();
        });
    }

    handleFileSelect(file) {
        if (!file) return;

        // 验证文件类型
        const validTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/mkv', 
                           'video/wmv', 'video/x-flv', 'video/webm'];
        
        if (!validTypes.includes(file.type)) {
            this.showError('不支持的文件格式。请选择MP4, AVI, MOV, MKV, WMV, FLV或WEBM格式的视频文件。');
            return;
        }

        // 验证文件大小 (100MB)
        const maxSize = 100 * 1024 * 1024;
        if (file.size > maxSize) {
            this.showError('文件大小超过限制。请选择小于100MB的视频文件。');
            return;
        }

        this.selectedFile = file;
        this.showFileInfo(file);
        this.uploadBtn.disabled = false;
    }

    showFileInfo(file) {
        const fileSize = this.formatFileSize(file.size);
        const fileType = file.type || '未知';
        
        this.fileDetails.innerHTML = `
            <p class="mb-1"><strong>文件名:</strong> ${file.name}</p>
            <p class="mb-1"><strong>文件大小:</strong> ${fileSize}</p>
            <p class="mb-0"><strong>文件类型:</strong> ${fileType}</p>
        `;
        
        this.fileInfo.style.display = 'block';
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async uploadVideo() {
        if (!this.selectedFile) return;

        const formData = new FormData();
        formData.append('file', this.selectedFile);

        this.uploadBtn.disabled = true;
        this.cancelBtn.style.display = 'inline-block';
        this.uploadProgress.style.display = 'block';

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`上传失败: ${response.status}`);
            }

            const result = await response.json();
            this.currentTaskId = result.task_id;
            
            this.showProcessingStatus(result.task_id);
            this.startPolling();

        } catch (error) {
            console.error('上传错误:', error);
            this.showError('上传失败: ' + error.message);
            this.resetUploadState();
        }
    }

    showProcessingStatus(taskId) {
        this.uploadProgress.style.display = 'none';
        this.processingStatus.style.display = 'block';
        this.taskId.textContent = `任务ID: ${taskId}`;
    }

    startPolling() {
        this.pollInterval = setInterval(async () => {
            try {
                const response = await fetch(`/api/result/${this.currentTaskId}`);
                const result = await response.json();

                if (result.status === 'completed') {
                    this.stopPolling();
                    this.showResult(result.result);
                } else if (result.status === 'failed') {
                    this.stopPolling();
                    this.showError(result.result?.error || '处理失败');
                }

            } catch (error) {
                console.error('轮询错误:', error);
            }
        }, 2000); // 每2秒检查一次
    }

    stopPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
    }

    showResult(result) {
        this.processingStatus.style.display = 'none';
        this.resultSection.style.display = 'block';

        // 显示主要结果
        const emotionIcon = this.emotionIcons[result.predicted_label] || '🤔';
        document.querySelector('.emotion-icon-large').textContent = emotionIcon;
        document.getElementById('emotionLabel').textContent = result.predicted_label;
        document.getElementById('confidenceScore').textContent = 
            `${(result.confidence * 100).toFixed(1)}%`;

        // 显示视频信息
        document.getElementById('fileName').textContent = 
            result.video_info?.filename || this.selectedFile.name;
        document.getElementById('totalFrames').textContent = 
            result.total_frames || '-';
        document.getElementById('processedFrames').textContent = 
            result.processed_frames || '-';
        document.getElementById('processingTime').textContent = 
            this.formatDate(result.processing_time) || '-';

        // 显示概率分布
        this.showProbabilityChart(result.probabilities);

        // 添加动画效果
        this.resultSection.classList.add('fade-in-up');
    }

    showProbabilityChart(probabilities) {
        const chartContainer = document.getElementById('probabilityChart');
        chartContainer.innerHTML = '';

        Object.entries(probabilities).forEach(([emotion, probability]) => {
            const percentage = (probability * 100).toFixed(1);
            const emotionIcon = this.emotionIcons[emotion] || '🤔';
            
            const barElement = document.createElement('div');
            barElement.className = 'probability-bar';
            barElement.innerHTML = `
                <div class="probability-label">
                    <span>${emotionIcon} ${emotion}</span>
                    <span>${percentage}%</span>
                </div>
                <div class="probability-progress">
                    <div class="probability-fill" style="width: ${percentage}%"></div>
                </div>
            `;
            
            chartContainer.appendChild(barElement);
        });
    }

    showError(message) {
        this.processingStatus.style.display = 'none';
        this.resultSection.style.display = 'none';
        this.errorSection.style.display = 'block';
        document.getElementById('errorMessage').textContent = message;
        this.resetUploadState();
    }

    cancelUpload() {
        this.stopPolling();
        this.resetUploadState();
    }

    resetUploadState() {
        this.uploadBtn.disabled = this.selectedFile ? false : true;
        this.cancelBtn.style.display = 'none';
        this.uploadProgress.style.display = 'none';
        this.processingStatus.style.display = 'none';
        this.progressBar.style.width = '0%';
        this.progressText.textContent = '';
    }

    formatDate(dateString) {
        if (!dateString) return '-';
        try {
            const date = new Date(dateString);
            return date.toLocaleString('zh-CN');
        } catch {
            return dateString;
        }
    }
}

// 保存结果功能
function saveResult() {
    const result = {
        emotion: document.getElementById('emotionLabel').textContent,
        confidence: document.getElementById('confidenceScore').textContent,
        fileName: document.getElementById('fileName').textContent,
        timestamp: new Date().toLocaleString('zh-CN')
    };

    // 保存到本地存储
    const savedResults = JSON.parse(localStorage.getItem('fer_results') || '[]');
    savedResults.unshift(result);
    
    // 只保留最近50个结果
    if (savedResults.length > 50) {
        savedResults.splice(50);
    }
    
    localStorage.setItem('fer_results', JSON.stringify(savedResults));
    
    // 显示成功消息
    const toast = document.createElement('div');
    toast.className = 'alert alert-success position-fixed top-0 end-0 m-3';
    toast.style.zIndex = '9999';
    toast.innerHTML = `
        <i class="fas fa-check-circle me-2"></i>
        结果已保存到本地
    `;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 3000);
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    new VideoUploader();
}); 