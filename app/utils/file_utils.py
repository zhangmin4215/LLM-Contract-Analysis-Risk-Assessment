import os
from werkzeug.utils import secure_filename
from config import Config
import logging

# 配置日志
logging.basicConfig(
    filename=Config.LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def allowed_file(filename):
    """
    检查文件扩展名是否允许
    :param filename: 文件名
    :return: 是否允许
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def save_uploaded_file(file):
    """
    保存上传的文件
    :param file: 上传的文件对象
    :return: 保存的文件路径
    """
    try:
        # 确保上传目录存在
        if not os.path.exists(Config.UPLOAD_FOLDER):
            os.makedirs(Config.UPLOAD_FOLDER)
        
        # 生成安全的文件名
        filename = secure_filename(file.filename)
        save_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        
        # 保存文件
        file.save(save_path)
        logging.info(f"文件保存成功: {save_path}")
        return save_path
    except Exception as e:
        logging.error(f"文件保存失败: {str(e)}")
        raise ValueError(f"文件保存失败: {str(e)}")

def cleanup_uploaded_files():
    """
    清理上传目录中的文件
    """
    try:
        for filename in os.listdir(Config.UPLOAD_FOLDER):
            file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                logging.info(f"清理文件: {file_path}")
    except Exception as e:
        logging.error(f"清理文件失败: {str(e)}")
        raise ValueError(f"清理文件失败: {str(e)}")

def get_file_extension(filename):
    """
    获取文件扩展名
    :param filename: 文件名
    :return: 文件扩展名（小写）
    """
    return filename.rsplit('.', 1)[1].lower()

def is_pdf_file(filename):
    """
    检查文件是否为PDF格式
    :param filename: 文件名
    :return: 是否为PDF
    """
    return get_file_extension(filename) == 'pdf'

def is_image_file(filename):
    """
    检查文件是否为图片格式
    :param filename: 文件名
    :return: 是否为图片
    """
    return get_file_extension(filename) in {'png', 'jpg', 'jpeg'}
