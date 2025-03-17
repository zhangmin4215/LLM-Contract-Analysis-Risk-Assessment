import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, Dataset
from config import Config
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 加载数据集
def load_and_preprocess_data(data_path):
    """
    加载并预处理合同数据集
    :param data_path: 数据集路径
    :return: 处理后的数据集
    """
    try:
        # 假设数据集是一个JSON文件，格式为：
        # [{"text": "合同文本", "info": {"party_a": "甲方", "party_b": "乙方", ...}}]
        dataset = load_dataset("json", data_files=data_path)
        
        # 将数据集转换为模型输入格式
        def preprocess_function(examples):
            # 构造提示词和输出
            inputs = [
                f"请从以下合同文本中提取关键信息：\n{text}\n"
                f"提取以下字段：\n"
                f"- 甲方名称（party_a）\n"
                f"- 乙方名称（party_b）\n"
                f"- 合同金额（amount）\n"
                f"- 签约日期（sign_date）\n"
                f"- 合同有效期（validity_period）\n"
                f"- 重要条款（key_terms）\n"
                for text in examples["text"]
            ]
            outputs = [
                f"甲方名称：{info['party_a']}\n"
                f"乙方名称：{info['party_b']}\n"
                f"合同金额：{info['amount']}\n"
                f"签约日期：{info['sign_date']}\n"
                f"合同有效期：{info['validity_period']}\n"
                f"重要条款：{info['key_terms']}\n"
                for info in examples["info"]
            ]
            return {"input": inputs, "output": outputs}
        
        # 应用预处理函数
        dataset = dataset.map(preprocess_function, batched=True)
        return dataset
    except Exception as e:
        logger.error(f"数据加载失败: {str(e)}")
        raise

# 微调模型
def finetune_llm(model_name, dataset, output_dir):
    """
    微调LLM模型
    :param model_name: 预训练模型名称
    :param dataset: 数据集
    :param output_dir: 模型保存路径
    """
    try:
        # 加载预训练模型和分词器
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        
        # 将输入和输出拼接为模型输入
        def tokenize_function(examples):
            combined_text = [f"{inp}\n{out}" for inp, out in zip(examples["input"], examples["output"])]
            return tokenizer(combined_text, truncation=True, padding="max_length", max_length=512)
        
        # 对数据集进行分词
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            save_steps=500,
            save_total_limit=2,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_steps=500,
            fp16=torch.cuda.is_available(),
            push_to_hub=False
        )
        
        # 初始化Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        
        # 开始训练
        logger.info("开始微调模型...")
        trainer.train()
        
        # 保存模型
        logger.info("保存微调后的模型...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"模型已保存至: {output_dir}")
    
    except Exception as e:
        logger.error(f"模型微调失败: {str(e)}")
        raise

# 主函数
def main():
    # 数据集路径
    data_path = "data/processed/contracts.json"
    
    # 模型名称
    model_name = Config.LLM_MODEL_NAME
    
    # 输出目录
    output_dir = "models/finetuned_llm"
    
    # 加载数据
    logger.info("加载数据集...")
    dataset = load_and_preprocess_data(data_path)
    
    # 微调模型
    logger.info("开始微调LLM模型...")
    finetune_llm(model_name, dataset, output_dir)

if __name__ == "__main__":
    main()
