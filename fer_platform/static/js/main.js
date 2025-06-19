// 主要的JavaScript功能
document.addEventListener('DOMContentLoaded', function() {
    // 通用功能初始化
    initializeCommonFeatures();
});

function initializeCommonFeatures() {
    // 工具提示初始化
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // 添加页面加载动画
    setTimeout(function() {
        document.body.classList.add('loaded');
    }, 100);
}

// 情感色彩映射
const emotionColors = {
    '快乐': '#28a745',
    '悲伤': '#007bff', 
    '愤怒': '#dc3545',
    '惊讶': '#ffc107',
    '恐惧': '#6f42c1',
    '厌恶': '#fd7e14',
    '中性': '#6c757d'
};

// 获取情感对应的颜色
function getEmotionColor(emotion) {
    return emotionColors[emotion] || '#6c757d';
}

// 显示加载状态
function showLoading(element, message = '处理中...') {
    element.innerHTML = `
        <div class="text-center">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <div class="mt-2">${message}</div>
        </div>
    `;
}

// 显示错误信息
function showError(element, message) {
    element.innerHTML = `
        <div class="alert alert-danger" role="alert">
            <i class="fas fa-exclamation-triangle"></i> ${message}
        </div>
    `;
}

// 显示成功信息
function showSuccess(element, message) {
    element.innerHTML = `
        <div class="alert alert-success" role="alert">
            <i class="fas fa-check-circle"></i> ${message}
        </div>
    `;
} 