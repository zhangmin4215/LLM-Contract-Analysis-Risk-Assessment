## LLM-Contract-Analysis-Risk-Assessment
A system for extracting key points and assessing risks in contracts using LLM.

### 一.项目简介  
智能合同分析系统是一个基于OCR（光学字符识别）和LLM（大语言模型）的自动化工具，旨在从扫描或拍照的合同文档中提取关键信息（如甲方、乙方、金额、日期等），并对合同内容进行风险评估。该系统可广泛应用于金融、法律、房地产等行业，提升合同处理效率并降低人工成本。  

### 二.功能特性  
1.合同信息提取
*支持PDF、PNG、JPG等格式的合同文档。
*自动提取甲方、乙方、金额、日期、有效期等关键信息。




### 三.安装与运行
#### 环境要求
*Python 3.8+  
*CUDA（可选，用于GPU加速）  
*Docker（可选，用于容器化部署）  
#### 安装步骤  
1.克隆项目仓库：  
```git clone https://github.com/yourusername/contract_analysis_system.git```  
```cd contract_analysis_system```  

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

### 项目结构

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




