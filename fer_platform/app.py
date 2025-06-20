"""
动态表情识别平台主应用
基于FastAPI的Web服务，提供视频上传、实时识别等功能
"""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from pathlib import Path
import os

from fer_platform.api.endpoints import router as api_router

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fer_platform.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="动态表情识别平台",
    description="基于MICACL的实时动态表情识别平台",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该限制具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 设置静态文件和模板
current_dir = Path(__file__).parent
static_dir = current_dir / "static"
templates_dir = current_dir / "templates"

# 确保目录存在
static_dir.mkdir(exist_ok=True)
templates_dir.mkdir(exist_ok=True)

# 挂载静态文件
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
app.mount("/css", StaticFiles(directory=str(static_dir / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(static_dir / "js")), name="js")
app.mount("/images", StaticFiles(directory=str(static_dir / "images")), name="images")

# 模板引擎
templates = Jinja2Templates(directory=str(templates_dir))

# 注册API路由
app.include_router(api_router, prefix="/api", tags=["API"])


@app.get("/", response_class=HTMLResponse, summary="主页")
async def read_root(request: Request):
    """主页"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/upload", response_class=HTMLResponse, summary="视频上传页面")
async def upload_page(request: Request):
    """视频上传页面"""
    return templates.TemplateResponse("upload.html", {"request": request})


@app.get("/realtime", response_class=HTMLResponse, summary="实时识别页面")
async def realtime_page(request: Request):
    """实时识别页面"""
    return templates.TemplateResponse("realtime.html", {"request": request})


@app.get("/results", response_class=HTMLResponse, summary="结果查看页面")
async def results_page(request: Request):
    """结果查看页面"""
    return templates.TemplateResponse("results.html", {"request": request})


@app.get("/about", response_class=HTMLResponse, summary="关于页面")
async def about_page(request: Request):
    """关于页面"""
    return templates.TemplateResponse("about.html", {"request": request})


@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("动态表情识别平台启动中...")
    
    # 导入配置检查模型状态
    from fer_platform.config import config
    model_info = config.get_model_info()
    
    logger.info("="*50)
    logger.info("平台配置信息")
    logger.info("="*50)
    logger.info(f"模型目录: {model_info['model_dir']}")
    logger.info(f"预训练模型: {'已加载' if model_info['has_pretrained'] else '未找到'}")
    if model_info['has_pretrained']:
        logger.info(f"模型文件: {model_info['model_path']}")
    else:
        logger.info(f"支持的模型文件: {', '.join(model_info['supported_files'])}")
        logger.info("提示: 请将预训练模型文件放在models目录中以获得更好的识别效果")
    logger.info("="*50)
    
    logger.info("平台启动完成！")
    logger.info("访问 http://localhost:8000 查看主页")
    logger.info("访问 http://localhost:8000/docs 查看API文档")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("动态表情识别平台正在关闭...")


if __name__ == "__main__":
    # 开发环境配置
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 