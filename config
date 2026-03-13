# config.py
import os
import torch

# ===================== 基础配置 =====================
# 设备配置（自动识别GPU/CPU）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== 路径配置 =====================
# 音频目录（存放待处理的wav文件）
AUDIO_DIR = "audios_wav"
# 文本输出目录
OUTPUT_DIR = "audios_txt"
# 整合结果目录（含冲突检测报告/评估指标）
INTEGRATED_DIR = "integrated_results_semantic"

# ===================== 模型配置 =====================
# 语义模型存储路径（autodl-tmp为云服务器路径，本地试用可改为./models）
SEMANTIC_MODEL_PATH = "/root/autodl-tmp/all-MiniLM-L6-v2"  # 云服务器路径
# SEMANTIC_MODEL_PATH = "./models/all-MiniLM-L6-v2"         # 本地试用路径（注释掉上面，启用这个）

# Whisper模型大小（tiny/base/small/medium/large，small平衡速度和精度）
WHISPER_MODEL_SIZE = "small"

# ===================== API配置（必须替换为真实密钥！） =====================
# DeepSeek API密钥（请自行替换为有效密钥，不要提交到GitHub）
DEEPSEEK_API_KEY = "your_deepseek_api_key_here"
MODEL_NAME = "deepseek-chat"

# ===================== 冲突检测参数 =====================
# 生成准确性阈值
GENERATION_ACCURACY_THRESHOLD = 0.6
# 最小聚类大小
MIN_CLUSTER_SIZE = 2
# 最小来源阈值
MIN_SOURCE_THRESHOLD = 1
# 允许通用来源
ALLOW_GENERAL_SOURCE = True

# ===================== HuggingFace镜像/缓存配置 =====================
# 镜像站（解决国内下载问题）
HF_ENDPOINT = "https://hf-mirror.com"
# 模型缓存根目录（云服务器/本地可按需修改）
HF_CACHE_DIR = "/root/autodl-tmp/huggingface"  # 云服务器
# HF_CACHE_DIR = "./models/huggingface"         # 本地试用

# ===================== 评估配置 =====================
# 默认评估样本数量
DEFAULT_EVAL_SAMPLES = 3
# 最大下载重试次数
MAX_DOWNLOAD_RETRIES = 3
