## LLM-Contract-Analysis-Risk-Assessment
A system for extracting key points and assessing risks in contracts using LLM.

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
