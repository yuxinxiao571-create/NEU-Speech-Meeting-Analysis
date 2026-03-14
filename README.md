# NEU-Conflict-Detection-System
东北大学 - 会议语音语义级冲突检测系统，支持语音转写、说话人分离、结构化分析、语义级冲突检测，输出Word/Excel可视化报告，支持真实音频处理和模拟数据评估。

## 项目功能
1. 🎧 语音处理：基于Whisper的语音转写 + Pyannote说话人分离，生成带时间戳的结构化对话
2. 📋 结构化分析：调用DeepSeek大模型，基于原始对话生成保真度100%的结构化会议分析
3. 🔍 语义冲突检测：自适应聚类 + 增强型LLM判定，精准识别会议中的对立/互斥观点
4. 📄 多格式输出：生成带冲突标注的Word报告、结构化Excel表格、可视化评估指标图
5. 📊 自动评估：支持模拟数据生成，计算生成质量、冲突检测性能、处理效率三类指标

## 快速开始（组内试用步骤）
### 前置条件
- Python 3.9~3.11（推荐3.10，避免依赖兼容问题）
- CUDA（可选，GPU加速，处理速度提升5-10倍）
- DeepSeek API密钥（获取地址：https://platform.deepseek.com/）
- 云服务器/本地环境（云服务器推荐autodl，已适配路径）

### 模型准备（关键！离线环境必看）
本项目已将核心模型预下载到本地缓存，无需在线下载：
1. **说话人分离模型**：`models--pyannote--speaker-diarization-3.1` `models--pyannote--segmentation-3.0` `models--pyannote--wespeaker-voxceleb-resnet34-LM`
   - 本地缓存路径：`/root/autodl-tmp/huggingface/hub/`  
   - 已启用 `HF_HUB_OFFLINE=1` 强制离线模式，无需联网验证
2. **语义模型**：`all-MiniLM-L6-v2`  
   - 云服务器路径：`/root/autodl-tmp/all-MiniLM-L6-v2`  
   - 本地路径：`./models/all-MiniLM-L6-v2`（需手动下载放到该目录）
3. **Whisper模型**：首次运行会自动下载到 `HF_CACHE_DIR` 配置的缓存目录（默认 `/root/autodl-tmp/`）

⚠️ 注意：若模型加载失败，请检查：
- 缓存目录是否存在对应模型文件
- `config.py` 中 `HF_CACHE_DIR`/`SEMANTIC_MODEL_PATH` 配置是否正确
- 已设置 `HF_HUB_OFFLINE=1`，禁止修改该环境变量
  
### 步骤1：克隆仓库
```bash
git clone https://github.com/你的GitHub用户名/NEU-Conflict-Detection-System.git
cd NEU-Conflict-Detection-System
```

### 步骤 2：安装依赖
```bash
# （推荐）创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# 一键安装所有依赖
pip install -r requirements.txt
```

### 步骤 3：配置关键信息
* 打开 config.py 文件

* 填写 DEEPSEEK_API_KEY：替换为你的 DeepSeek 有效 API 密钥

* （可选）调整路径配置：
  * 云服务器：保持默认路径（/root/autodl-tmp/）
  * 本地试用：注释云服务器路径，启用本地路径（./models/）

### 步骤 4：准备测试音频（处理真实音频模式）
* 将待处理的 WAV 格式音频文件放入 audios_wav 目录
  
* 音频要求：单声道、采样率 16kHz（非必须，代码会自动预处理）
  
* 推荐音频时长：1-5 分钟（首次测试快速验证）
  
### 步骤 5：运行项目
```bash
python main.py
```

### 步骤 6：选择运行模式
#### 模式 1（处理真实音频）
自动处理 audios_wav 中的所有 WAV 文件，生成：

* Word 报告：integrated_results_semantic/[音频名]_report.docx（带冲突标注）
  
* Excel 报告：integrated_results_semantic/[音频名]_report.xlsx（结构化数据）
  
* 冲突日志：integrated_results_semantic/conflict_detection_logs.json
  
#### 模式 2（模拟数据评估）
生成模拟对话数据，自动评估系统性能，输出：

* 评估指标：控制台打印平均准确率 / 召回率 / F1 值
  
* 可视化图：integrated_results_semantic/evaluation_metrics.png
  
### 核心配置说明
| 配置项 | 说明 | 推荐值 |
|--------|------|--------|
| DEVICE | 运行设备 | cuda（GPU）/ cpu（CPU） |
| WHISPER_MODEL_SIZE | Whisper 模型大小 | small（平衡速度 / 精度） |
| DEEPSEEK_API_KEY | DeepSeek API 密钥 | 自行申请的有效密钥 |
| SEMANTIC_MODEL_PATH | 语义模型路径 | 云服务器：/root/autodl-tmp/all-MiniLM-L6-v2；本地：./models/all-MiniLM-L6-v2 |
| GENERATION_ACCURACY_THRESHOLD | 生成准确性阈值 | 0.6 |
| MIN_CLUSTER_SIZE | 最小聚类大小 | 2 |
| `HF_CACHE_DIR` | 模型缓存根目录 | /root/autodl-tmp/ |
| `HF_ENDPOINT` | HF 镜像源（解决下载问题） | https://hf-mirror.com |
| `HF_HUB_OFFLINE` | 强制离线模式 | 1（启用） |

## 注意事项
1. 模型下载：**说话人分离/语义模型已预下载到本地**，仅Whisper模型首次运行自动下载（约2-5GB），网络较慢时请耐心等待。
   
2. 密钥安全：不要将真实的 `DEEPSEEK_API_KEY` 提交到 GitHub，保持 `config.py` 中的占位符。
   
3. 离线模式：代码默认启用 HF 离线模式，需确保模型已下载到缓存目录（`/root/autodl-tmp/huggingface/hub/`）。
   
4. 音频格式：优先使用 WAV 格式，MP3 格式需自行转换（可使用 `librosa` 转换）。
   
5. 性能优化：处理长音频（>10 分钟）时，建议使用 GPU，CPU 处理时间会显著增加。
    
6. 模型加载失败排查：
   - 检查 `config.py` 中 `SEMANTIC_MODEL_PATH` 是否指向正确的本地模型路径；
   - 确认 `pyannote/speaker-diarization-3.1` 等已放到 `/root/autodl-tmp/huggingface/hub/` 目录。
   
### 项目结构
```plaintext
NEU-Conflict-Detection-System/
├── audios_wav/                 # 测试音频目录（用户自行放入WAV文件）
├── audios_txt/                 # 文本输出目录（自动生成）
├── integrated_results_semantic/# 整合结果目录（报告/评估指标）
├── config.py                   # 集中配置文件（仅需修改此处）
├── main.py                     # 核心代码（一键运行）
├── requirements.txt            # 依赖清单（一键安装）
└── README.md                   # 项目说明文档（组内试用指南）
```
