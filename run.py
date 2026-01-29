# run.py
import os
import sys
import pandas as pd

# 把src目录加入Python路径，支持模块导入
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# 导入配置和模块
from config import *
from data_preprocessor import DataPreprocessor
from speech_processor import SpeechProcessor
from structured_generator import StructuredGenerator
from conflict_detector import ConflictDetector
from result_presenter import ResultPresenter

def create_output_dir():
    """自动创建输出目录，避免文件不存在报错"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"输出目录已创建/验证：{OUTPUT_DIR}")

def check_input_audio():
    """检查输入音频文件是否存在"""
    if not os.path.exists(INPUT_AUDIO_PATH):
        raise FileNotFoundError(f"输入音频文件不存在：{INPUT_AUDIO_PATH}\n请在sample_data目录下放入名为sample_meeting.wav的测试音频")
    print(f"输入音频已验证：{INPUT_AUDIO_PATH}")

def check_config():
    """检查配置文件中的关键参数是否填写"""
    if PYANNOTE_TOKEN == "your_huggingface_token_here":
        raise ValueError("请先在config.py中填写有效的Pyannote Token（Hugging Face获取）")
    if LLM_API_KEY == "your_openai_api_key_here":
        raise ValueError("请先在config.py中填写有效的OpenAI API密钥")
    print("配置文件验证通过")

def main():
    """项目主流程：串联所有模块，一键运行"""
    try:
        # 步骤1：初始化与前置检查
        print("=" * 60)
        print("【步骤1/5】项目初始化与前置检查...")
        create_output_dir()
        check_input_audio()
        check_config()
        print("=" * 60)

        # 步骤2：音频预处理
        print("【步骤2/5】音频预处理（格式转换+静音去除+降噪）...")
        preprocessor = DataPreprocessor()
        processed_audio = preprocessor.process_audio(INPUT_AUDIO_PATH, PROCESSED_AUDIO_PATH)
        print(f"音频预处理完成，处理后音频保存至：{processed_audio}")
        print("=" * 60)

        # 步骤3：语音转写与说话人对齐
        print("【步骤3/5】语音转写与说话人分离对齐...")
        speech_processor = SpeechProcessor(PYANNOTE_TOKEN)
        # 转写
        transcribe_df = speech_processor.transcribe_audio(processed_audio)
        # 说话人分离
        speaker_df = speech_processor.separate_speakers(processed_audio)
        # 对齐
        aligned_df = speech_processor.align_speech_text(transcribe_df, speaker_df)
        # 保存对齐结果
        aligned_df.to_csv(ALIGNED_CSV_PATH, index=False, encoding="utf-8")
        print(f"语音处理完成，对齐结果保存至：{ALIGNED_CSV_PATH}（共{len(aligned_df)}条有效记录）")
        print("=" * 60)

        # 步骤4：结构化报告生成与来源校验
        print("【步骤4/5】结构化报告生成与来源校验...")
        generator = StructuredGenerator(LLM_API_KEY, LLM_MODEL)
        # 格式化输入
        formatted_input = generator.format_input(aligned_df)
        # 生成结构化报告
        structured_text = generator.generate_structured(formatted_input)
        # 来源校验
        valid_structured_text = generator.verify_source(structured_text, aligned_df)
        # 保存结构化报告
        with open(STRUCTURED_TXT_PATH, "w", encoding="utf-8") as f:
            f.write(valid_structured_text)
        print(f"结构化报告生成完成，保存至：{STRUCTURED_TXT_PATH}")
        print("=" * 60)

        # 步骤5：冲突检测与结果导出
        print("【步骤5/5】冲突检测与完整报告生成...")
        # 冲突检测
        detector = ConflictDetector()
        text_list = aligned_df["text"].tolist()
        # 关键词过滤
        conflict_candidates = detector.keyword_filter(text_list)
        candidate_texts = [text for _, text in conflict_candidates]
        conflicts = []

        if len(candidate_texts) >= 2:
            # 语义聚类
            cluster_sentences = detector.semantic_clustering(candidate_texts)
            # 深度冲突检测
            for i in range(len(cluster_sentences)):
                for j in range(i+1, len(cluster_sentences)):
                    is_conflict, prob = detector.detect_conflict(cluster_sentences[i], cluster_sentences[j])
                    if is_conflict:
                        conflicts.append({
                            "text1": cluster_sentences[i],
                            "text2": cluster_sentences[j],
                            "conflict_prob": prob
                        })
            # 保存冲突结果
            pd.DataFrame(conflicts).to_csv(CONFLICT_CSV_PATH, index=False, encoding="utf-8")
            print(f"冲突检测完成，共检测到{len(conflicts)}条冲突观点，保存至：{CONFLICT_CSV_PATH}")
        else:
            print("无足够候选文本进行冲突检测，跳过该步骤")

        # 生成Word报告
        presenter = ResultPresenter()
        word_doc = presenter.generate_word_report(valid_structured_text, conflicts)
        word_doc.save(WORD_REPORT_PATH)
        print(f"Word完整报告生成完成，保存至：{WORD_REPORT_PATH}")

        # 计算评估指标
        metrics = presenter.calculate_metrics(true_conflicts=[], pred_conflicts=conflicts)
        pd.DataFrame([metrics]).to_excel(METRICS_EXCEL_PATH, index=False, engine="openpyxl")
        print(f"评估指标计算完成，保存至：{METRICS_EXCEL_PATH}")
        print("=" * 60)

        # 运行完成提示
        print("✅ 项目运行全部完成！所有结果已保存至output目录")
        print(f"📄 核心结果文件：")
        print(f"   - 对齐数据：{ALIGNED_CSV_PATH}")
        print(f"   - 结构化报告（纯文本）：{STRUCTURED_TXT_PATH}")
        print(f"   - 完整Word报告：{WORD_REPORT_PATH}")
        print(f"   - 评估指标：{METRICS_EXCEL_PATH}")

    except Exception as e:
        print(f"❌ 项目运行失败：{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()