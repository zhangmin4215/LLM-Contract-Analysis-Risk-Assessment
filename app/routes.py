from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from .models.ocr_model import OCRProcessor
from .models.llm_model import LLMExtractor
from .models.risk_assessment import RiskAssessor
from config import Config
import os
import logging
from datetime import datetime

app = Flask(__name__)
app.config.from_object(Config)

# 初始化处理器
ocr_processor = OCRProcessor()
llm_extractor = LLMExtractor()
risk_assessor = RiskAssessor()

# 配置日志
logging.basicConfig(
    filename=Config.LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 检查文件上传
        if 'file' not in request.files:
            return jsonify({"error": "未检测到文件上传"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "未选择文件"}), 400
        
        if not ocr_processor.allowed_file(file.filename):
            return jsonify({"error": "文件类型不支持"}), 400
        
        try:
            # 保存文件
            filename = secure_filename(f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}")
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            
            # OCR处理
            logging.info(f"开始处理文件: {filename}")
            if filename.lower().endswith('.pdf'):
                text = ocr_processor.process_pdf(save_path)
            else:
                text = ocr_processor.process_image(save_path)
            
            # 信息提取
            contract_info = llm_extractor.extract_contract_info(text)
            
            # 风险评估
            risk_assessment = risk_assessor.assess_risk(text)
            
            return jsonify({
                "status": "success",
                "contract_info": contract_info,
                "risk_assessment": risk_assessment,
                "text_snippet": text[:500] + "..."  # 返回部分文本预览
            })
            
        except Exception as e:
            logging.error(f"处理失败: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    return render_template('index.html')
