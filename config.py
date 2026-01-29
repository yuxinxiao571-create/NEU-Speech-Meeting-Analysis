# config.py
import os

# ---------------------- 路径配置（无需修改，自动适配） ----------------------
# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# 输入音频路径（默认使用sample_data下的测试音频）
INPUT_AUDIO_PATH = os.path.join(PROJECT_ROOT, "sample_data", "sample_meeting.wav")
# 输出目录（自动创建）
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
# 处理后音频路径
PROCESSED_AUDIO_PATH = os.path.join(OUTPUT_DIR, "processed_audio.wav")
# 各类结果输出路径
ALIGNED_CSV_PATH = os.path.join(OUTPUT_DIR, "aligned_speech_text.csv")
CONFLICT_CSV_PATH = os.path.join(OUTPUT_DIR, "conflict_results.csv")
STRUCTURED_TXT_PATH = os.path.join(OUTPUT_DIR, "structured_report.txt")
WORD_REPORT_PATH = os.path.join(OUTPUT_DIR, "会议分析报告.docx")
METRICS_EXCEL_PATH = os.path.join(OUTPUT_DIR, "评估指标.xlsx")

# ---------------------- 模型配置（可根据需求微调） ----------------------
# Whisper模型选择（tiny/base/small/medium/large，medium平衡速度和精度）
WHISPER_MODEL = "medium"
# Pyannote说话人分离模型
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"
# BERT模型（语义校验/冲突检测）
BERT_MODEL = "bert-base-uncased"
# 冲突检测概率阈值
CONFLICT_THRESHOLD = 0.8
# 语义相似度校验阈值（来源校验）
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
# 语义聚类簇数
CLUSTER_NUM = 3

# ---------------------- 授权配置（必须修改！填写有效密钥） ----------------------
# Hugging Face Token（获取地址：https://huggingface.co/settings/tokens）
# 注意：需先同意Pyannote模型协议：https://huggingface.co/pyannote/speaker-diarization-3.1
PYANNOTE_TOKEN = "your_huggingface_token_here"

# LLM API配置（GPT/Deepseek，选择其一填写）
# 1. OpenAI GPT（获取地址：https://platform.openai.com/api-keys）
LLM_API_KEY = "your_openai_api_key_here"
LLM_MODEL = "gpt-3.5-turbo"  # 推荐先用gpt-3.5-turbo测试（成本低、速度快），可替换为gpt-4o