from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from config import Config
import json
import re

class LLMExtractor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            Config.LLM_MODEL_NAME, 
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            Config.LLM_MODEL_NAME,
            trust_remote_code=True
        ).to(Config.LLM_DEVICE)
        
    def extract_contract_info(self, text):
        """从合同文本中提取结构化信息"""
        prompt = f"""
        你是一个专业合同分析助手，请从以下文本中提取关键信息：
        {text}
        
        需要提取的字段：
        - 甲方名称（party_a）
        - 乙方名称（party_b）
        - 合同金额（amount）
        - 签约日期（sign_date）
        - 合同有效期（validity_period）
        - 重要条款（key_terms）
        
        要求：
        1. 金额统一转换为人民币元
        2. 日期格式为YYYY-MM-DD
        3. 使用JSON格式返回
        """
        
        response = self._generate_response(prompt)
        return self._parse_json_response(response)
    
    def _generate_response(self, prompt, max_length=1024):
        """生成LLM响应"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(Config.LLM_DEVICE)
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=0.3,  # 降低随机性
            top_p=0.9
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _parse_json_response(self, text):
        """从文本中解析JSON结构"""
        try:
            # 使用正则表达式提取JSON部分
            json_str = re.search(r'\{.*\}', text, re.DOTALL).group()
            return json.loads(json_str)
        except Exception as e:
            raise ValueError(f"LLM响应解析失败: {str(e)}")
