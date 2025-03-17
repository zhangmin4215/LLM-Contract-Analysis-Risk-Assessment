import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import Config

class RiskAssessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(Config.RISK_MODEL_PATH)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            Config.RISK_MODEL_PATH
        ).to(Config.LLM_DEVICE)
    
    def assess_risk(self, text):
        """评估合同风险"""
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(Config.LLM_DEVICE)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 假设模型输出0-1之间的风险概率
        risk_score = torch.sigmoid(outputs.logits).item()
        return {
            "risk_level": "高风险" if risk_score > 0.7 else "中等风险" if risk_score > 0.4 else "低风险",
            "risk_score": round(risk_score, 3)
        }
