# 使用官方 Python 3.9 镜像作为基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \  # 用于 OpenCV 和其他图像处理库
    libglib2.0-0 \    # 用于 PaddleOCR
    poppler-utils \   # 用于 PDF 处理
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件到容器中
COPY . .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 设置环境变量
ENV FLASK_APP=app/routes.py
ENV FLASK_ENV=production
ENV UPLOAD_FOLDER=/app/uploads

# 创建上传目录
RUN mkdir -p ${UPLOAD_FOLDER}

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app.routes:app"]
