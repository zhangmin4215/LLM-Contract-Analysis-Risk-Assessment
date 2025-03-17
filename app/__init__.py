from flask import Flask
from config import Config
import os

def create_app():
    """
    创建并配置Flask应用
    """
    # 初始化Flask应用
    app = Flask(__name__)
    
    # 加载配置
    app.config.from_object(Config)
    
    # 确保上传文件夹存在
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # 注册路由
    from .routes import app as routes_blueprint
    app.register_blueprint(routes_blueprint)
    
    return app
