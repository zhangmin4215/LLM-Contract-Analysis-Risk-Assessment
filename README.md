## LLM-Contract-Analysis-Risk-Assessment
A system for extracting key points and assessing risks in contracts using LLM.

### 一.项目简介  
智能合同分析系统是一个基于OCR（光学字符识别）和LLM（大语言模型）的自动化工具，旨在从扫描或拍照的合同文档中提取关键信息（如甲方、乙方、金额、日期等），并对合同内容进行风险评估。该系统可广泛应用于金融、法律、房地产等行业，提升合同处理效率并降低人工成本。  
### 二.功能特性  
1.合同信息提取
* 支持PDF、PNG、JPG等格式的合同文档。  
* 自动提取甲方、乙方、金额、日期、有效期等关键信息。  
* 输出结构化数据（JSON格式）。  

2.风险评估  
* 基于LLM的语义分析，识别合同中的潜在风险。  
* 提供风险等级（低风险、中等风险、高风险）和风险评分。

3.用户友好界面  
* 提供Web界面，支持文件上传和结果展示。  
* 实时显示提取信息和风险评估结果。

4.高性能与可扩展性  
* 支持GPU加速，提升处理速度。
* 模块化设计，方便扩展新功能。

### 三.安装与运行
#### 环境要求
* Python 3.8+  
* CUDA（可选，用于GPU加速）  
* Docker（可选，用于容器化部署）  
#### 安装步骤  
1.克隆项目仓库：  
```git clone https://github.com/zhangmin4215/LLM-Contract-Analysis-Risk-Assessment.git```  

```cd LLM-Contract-Analysis-Risk-Assessment```  

2.安装依赖  
```pip install -r requirements.txt```  

3.下载模型文件（可选）  
如果需要使用自定义模型，请将模型文件放置在models/目录下。  
默认使用PaddleOCR和ChatGLM-6B模型。  

4.启动服务  
```export FLASK_APP=app/routes.py```  

```flask run --host=0.0.0.0 --port=5000```  

5.访问Web界面  
打开浏览器，访问http://localhost:5000

### 四.使用方法  
#### 1.上传合同文件  
* 点击“选择文件”按钮，上传PDF、PNG或JPG格式的合同文档。
* 点击“分析合同”按钮，开始处理。

#### 2.查看结果
* 提取的关键信息将显示在“提取信息”区域。
* 风险评估结果将显示在“风险评估”区域。

#### 3.下载结果
* 支持将提取信息和风险评估结果导出为JSON文件。

### 五.项目结构

LLM-Contract-Analysis-Risk-Assessment/  
│  
├── app/                  # Web应用主目录  
│   ├── __init__.py  
│   ├── routes.py         # Flask路由  
│   ├── models/           # 模型相关代码  
│   │   ├── ocr_model.py  
│   │   ├── llm_model.py  
│   │   └── risk_assessment.py  
│   ├── utils/            # 工具函数  
│   │   ├── file_utils.py  
│   │   └── logger.py  
│   ├── templates/        # 前端模板  
│   │   └── index.html  
│   └── static/           # 静态资源  
│       └── styles.css  
│  
├── data/                 # 数据集和预处理脚本  
│   ├── raw_contracts/    # 原始合同扫描件  
│   ├── processed/        # 处理后的文本数据  
│   └── preprocess.py     # 数据预处理脚本  
│  
├── training/             # 模型训练脚本  
│   ├── finetune_llm.py   # LLM微调脚本  
│   └── train_risk.py     # 风险评估模型训练  
│  
├── config.py             # 配置文件  
├── requirements.txt      # 依赖列表  
├── Dockerfile            # Docker部署文件  
└── README.md             # 项目说明  

### 六.模型训练
#### 微调LLM模型
1.准备训练数据  
* 将合同文本和对应的标注信息放置在 data/processed/ 目录下。

2.运行微调脚本  
```python training/finetune_llm.py```

3.保存模型  
* 微调后的模型将保存在models/目录下。

#### 训练风险评估模型
1.准备训练数据  
* 将合同文本和对应的风险标签放置在data/processed/目录下。

2.运行训练脚本  
```python training/train_risk.py```

3.保存模型
* 训练后的模型将保存在models/目录下。

### 七.部署
#### Docker部署
1.构建Docker镜像  
```docker build -t contract_analysis_system .```

2.运行容器  
```docker run -p 5000:5000 contract_analysis_system```

3.访问服务
* 打开浏览器，访问 http://localhost:5000

### 八.数据集
#### 1.公开合同数据集（适用于模型预训练与微调）
（1）Contract Understanding Atticus Dataset (CUAD)  
* 简介：由IBM和NLP研究团队发布，包含510份合同，标注了41种法律条款（如免责条款、自动续约、赔偿等）。
* 用途：适用于合同风险识别、条款提取、法律条款分类。
* 链接：https://huggingface.co/datasets/cuad

（2）EDGAR合同数据集  
* 简介：来自美国SEC的EDGAR数据库，包含大量上市公司披露的合同，如商业协议、并购协议、投资协议等。
* 用途：适用于金融、投资、并购相关合同解析。
* 链接：https://www.sec.gov/edgar.shtml

（3）LexNLP合同数据
* 简介：LexNLP是一个法律NLP处理工具，提供了开源合同数据和法律文本解析能力。
* 用途：适用于法律术语识别、合同结构化解析。
* 链接：https://github.com/LexPredict/lexpredict-lexnlp
