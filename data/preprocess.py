import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pytesseract
from pdf2image import convert_from_path
import re
from config import Config

class DataPreprocessor:
    def __init__(self, raw_data_dir, processed_data_dir):
        """
        初始化数据预处理器
        :param raw_data_dir: 原始数据目录
        :param processed_data_dir: 处理后的数据目录
        """
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        os.makedirs(self.processed_data_dir, exist_ok=True)

    def preprocess_images(self):
        """
        预处理图像数据（增强、二值化等）
        """
        for filename in os.listdir(self.raw_data_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(self.raw_data_dir, filename)
                processed_image = self._enhance_image(image_path)
                save_path = os.path.join(self.processed_data_dir, filename)
                processed_image.save(save_path)
                print(f"Processed image saved: {save_path}")

    def preprocess_pdfs(self):
        """
        预处理PDF文件（转换为图像并增强）
        """
        for filename in os.listdir(self.raw_data_dir):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(self.raw_data_dir, filename)
                images = convert_from_path(pdf_path)
                for i, image in enumerate(images):
                    processed_image = self._enhance_image(image)
                    save_path = os.path.join(
                        self.processed_data_dir,
                        f"{filename}_page_{i + 1}.png"
                    )
                    processed_image.save(save_path)
                    print(f"Processed PDF page saved: {save_path}")

    def _enhance_image(self, image):
        """
        图像增强（对比度、亮度、锐化等）
        :param image: 图像路径或PIL图像对象
        :return: 增强后的PIL图像
        """
        if isinstance(image, str):
            image = Image.open(image)
        
        # 增强对比度
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        # 增强亮度
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.2)
        
        # 转换为灰度图
        image = image.convert('L')
        
        # 二值化
        image_np = np.array(image)
        _, image_np = cv2.threshold(image_np, 128, 255, cv2.THRESH_BINARY)
        image = Image.fromarray(image_np)
        
        return image

    def extract_text_from_images(self):
        """
        从处理后的图像中提取文本并保存为TXT文件
        """
        for filename in os.listdir(self.processed_data_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(self.processed_data_dir, filename)
                text = self._extract_text(image_path)
                txt_path = os.path.join(
                    self.processed_data_dir,
                    f"{os.path.splitext(filename)[0]}.txt"
                )
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"Text extracted and saved: {txt_path}")

    def _extract_text(self, image_path):
        """
        使用Tesseract从图像中提取文本
        :param image_path: 图像路径
        :return: 提取的文本
        """
        text = pytesseract.image_to_string(image_path, lang='chi_sim+eng')
        return self._clean_text(text)

    def _clean_text(self, text):
        """
        清理提取的文本（去除多余空格、特殊字符等）
        :param text: 原始文本
        :return: 清理后的文本
        """
        # 去除多余的空格和换行
        text = re.sub(r'\s+', ' ', text).strip()
        # 去除特殊字符
        text = re.sub(r'[^\w\s.,，。]', '', text)
        return text

    def run(self):
        """
        运行数据预处理流程
        """
        print("Starting image preprocessing...")
        self.preprocess_images()
        print("Starting PDF preprocessing...")
        self.preprocess_pdfs()
        print("Starting text extraction...")
        self.extract_text_from_images()
        print("Data preprocessing completed!")


if __name__ == "__main__":
    # 配置路径
    raw_data_dir = os.path.join('data', 'raw_contracts')
    processed_data_dir = os.path.join('data', 'processed')
    
    # 初始化并运行预处理器
    preprocessor = DataPreprocessor(raw_data_dir, processed_data_dir)
    preprocessor.run()
