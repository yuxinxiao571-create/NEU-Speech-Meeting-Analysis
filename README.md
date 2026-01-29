# NEU-Speech-Meeting-Analysis
NEU-Speech-Meeting-Analysis/  # 项目根目录（仓库名，简洁贴合论文主题）
├── .gitignore                # git忽略文件，不提交无用内容
├── README.md                 # 核心说明文档，组内试用的入口指南
├── requirements.txt          # 依赖清单，一键安装
├── config.py                 # 配置文件，集中管理所有密钥、路径、参数
├── src/                      # 核心源代码目录（模块化拆分，更规范）
│   ├── __init__.py           # 标记为Python包
│   ├── data_preprocessor.py  # 数据预处理模块
│   ├── speech_processor.py   # 说话人分离&语音转写模块
│   ├── structured_generator.py # 结构化内容生成模块
│   ├── conflict_detector.py  # 冲突检测模块
│   └── result_presenter.py   # 结果展示&评估模块
├── run.py                    # 项目入口文件（一键运行，无需修改核心代码）
├── sample_data/              # 示例数据（放一个小型测试音频，方便组内试用）
│   └── sample_meeting.wav
└── output/                   # 输出目录（自动生成，存放运行结果）
