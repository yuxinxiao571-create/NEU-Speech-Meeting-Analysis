# NEU-Speech-Meeting-Analysis
东北大学本科毕业论文项目：会议语音结构化分析与冲突检测系统，支持语音转写、说话人分离、结构化报告生成、冲突观点检测，可直接用于组内会议记录分析。

## 项目功能
1. 🎧 音频预处理：格式转换、静音去除、简单降噪
2. 🗣️ 语音转写+说话人分离：基于Whisper+Pyannote，生成「时间-说话人-文本」结构化数据
3. 📋 结构化报告生成：基于大语言模型，生成含议题、专家意见、讨论结果的会议报告
4. 🔍 冲突检测：关键词过滤+语义聚类+BERT深度判定，识别会议中的冲突观点
5. 📄 结果导出：支持CSV、TXT、Word、Excel多种格式输出，方便后续分析

## 快速开始（组内试用步骤）
### 1. 克隆仓库
```bash
git clone https://github.com/yuxinxiao571-create/NEU-Speech-Meeting-Analysis.git
cd NEU-Speech-Meeting-Analysis

