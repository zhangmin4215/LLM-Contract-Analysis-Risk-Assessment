from paddleocr import PaddleOCR
from PIL import Image
import pdf2image
import numpy as np
import os
from config import Config

class OCRProcessor:
    def __init__(self):
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=Config.OCR_LANG,
            show_log=False  # 关闭冗余日志
        )
    
    def process_image(self, image_path):
        """处理单张图片"""
        result = self.ocr.ocr(image_path, cls=True)
        return self._format_result(result)
    
    def process_pdf(self, pdf_path):
        """处理PDF文档（转换为多张图片后识别）"""
        images = pdf2image.convert_from_path(pdf_path)
        full_text = []
        for i, image in enumerate(images):
            image_np = np.array(image)
            result = self.ocr.ocr(image_np)
            full_text.append(self._format_result(result))
        return "\n".join(full_text)
    
    def _format_result(self, result):
        """将OCR结果格式化为文本"""
        return " ".join([line[1][0] for line in result[0]])

    @staticmethod
    def allowed_file(filename):
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS
