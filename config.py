import os

class Config:
    # 文件上传配置
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'app', 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    
    # OCR模型配置
    OCR_MODEL_PATH = 'paddleocr'
    OCR_LANG = 'ch'  # 支持中英文混合
    
    # LLM模型配置
    LLM_MODEL_NAME = 'THUDM/chatglm-6b'
    LLM_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 风险评估模型路径
    RISK_MODEL_PATH = 'models/risk_assessment.pth'
    
    # 日志配置
    LOG_FILE = 'contract_analysis.log'
