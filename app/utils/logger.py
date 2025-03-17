import logging
import os
from logging.handlers import RotatingFileHandler
from config import Config

# 确保日志目录存在
log_dir = os.path.dirname(Config.LOG_FILE)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 配置日志格式
log_format = "%(asctime)s - %(levelname)s - %(message)s"
date_format = "%Y-%m-%d %H:%M:%S"

# 创建日志记录器
logger = logging.getLogger("contract_analysis")
logger.setLevel(logging.INFO)

# 文件日志处理器（按文件大小轮换）
file_handler = RotatingFileHandler(
    Config.LOG_FILE,
    maxBytes=10 * 1024 * 1024,  # 每个日志文件最大10MB
    backupCount=5,  # 保留5个备份文件
    encoding="utf-8"
)
file_handler.setFormatter(logging.Formatter(log_format, date_format))
logger.addHandler(file_handler)

# 控制台日志处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(log_format, date_format))
logger.addHandler(console_handler)

# 示例日志函数
def log_info(message):
    """记录INFO级别日志"""
    logger.info(message)

def log_error(message):
    """记录ERROR级别日志"""
    logger.error(message)

def log_warning(message):
    """记录WARNING级别日志"""
    logger.warning(message)

def log_debug(message):
    """记录DEBUG级别日志"""
    logger.debug(message)
