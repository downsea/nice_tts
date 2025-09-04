import os
import re
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from datetime import datetime

def load_llm_config():
    """
    Loads LLM configuration from .env files.
    It loads a global ~/.env file first, then a local .env file in the current
    directory, with local variables overriding global ones.
    """
    global_dotenv_path = Path.home() / ".env"
    if global_dotenv_path.is_file():
        load_dotenv(dotenv_path=global_dotenv_path)

    local_dotenv_path = find_dotenv(usecwd=True)
    if local_dotenv_path:
        load_dotenv(dotenv_path=local_dotenv_path, override=True)

    config = {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_API_BASE"),
        "model_name": os.getenv("OPENAI_MODEL_NAME"),
    }
    if not all(config.values()):
        raise ValueError(
            "Missing one or more required environment variables: "
            "OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_MODEL_NAME. "
            "Please create a .env file in your project directory or home directory (~/.env)."
        )
    return config

def _parse_srt_to_text(srt_content: str) -> str:
    """
    Parses SRT content to extract plain text.
    """
    lines = srt_content.strip().split('\n')
    text_parts = []
    for line in lines:
        line = line.strip()
        if line and not line.isdigit() and '-->' not in line:
            text_parts.append(line)
    return " ".join(text_parts)

def refine_transcript(srt_path: str, output_fine_path: str) -> str:
    """
    Refines a transcript from an SRT file using an LLM and saves it to a specified path.
    """
    p_srt_path = Path(srt_path)
    if not p_srt_path.is_file():
        raise FileNotFoundError(f"SRT file not found at: {srt_path}")

    config = load_llm_config()
    with open(p_srt_path, 'r', encoding='utf-8') as f:
        srt_content = f.read()

    transcript_text = _parse_srt_to_text(srt_content)
    if not transcript_text:
        print("警告: 未能从SRT文件中提取任何文本。")
        Path(output_fine_path).touch()
        return output_fine_path

    prompt = f"""
您是一位专业的速记稿后期处理专家。您的任务是优化下方由自动语音识别（ASR）生成的原始会议记录文本。请严格按照以下要求操作，以生成一份流畅、准确、易读的精校版中文文稿。

**处理要求：**
1.  **润色和修正**：修正文本中的错别字、语法错误，并确保语句通顺、表达准确。
2.  **优化标点符号**：根据上下文和語氣，正确使用中文标点符号，如逗号（，）、句号（。）、顿号（、）、问号（？）等。
3.  **去除冗余**：删除对话中的无意义口头禅、重复词语和语气词（例如：“嗯”、“呃”、“那个”、“就是说”等）。
4.  **合并与分段**：将碎片化句子合并为完整、连贯的句子。根据内容的逻辑关系，将文本合理地划分为段落。
5.  **保持原意**：必须最大限度地保留原始对话的意图和信息。
6.  **格式要求**：最终输出结果应为纯净的、连续的中文文本段落。**不要**添加任何标题、说话人标识或任何形式的标记。

**原始速记稿文本如下：**
---
{transcript_text}
---
"""

    print("正在连接大语言模型以优化文本...")
    client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
    response = client.chat.completions.create(
        model=config["model_name"],
        messages=[
            {"role": "system", "content": "你是一位专业的速记稿后期处理专家，请使用中文输出。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    refined_text = response.choices[0].message.content or ""

    with open(output_fine_path, 'w', encoding='utf-8') as f:
        f.write(refined_text)
    print(f"优化后的文本已保存至: {output_fine_path}")
    return output_fine_path

def summarize_transcript(
    refined_text_path: str, original_audio_path: str, output_md_path: str
) -> str:
    """
    Generates a Chinese meeting summary and saves it to a specified path.
    """
    p_refined_text_path = Path(refined_text_path)
    p_original_audio_path = Path(original_audio_path)
    if not p_refined_text_path.is_file():
        raise FileNotFoundError(f"Refined transcript file not found at: {refined_text_path}")

    config = load_llm_config()
    with open(p_refined_text_path, 'r', encoding='utf-8') as f:
        refined_text = f.read()

    if not refined_text.strip():
        print("警告: 优化后的文本为空，无法生成摘要。")
        Path(output_md_path).touch()
        return output_md_path

    # Create relative paths for the links in the markdown file
    output_dir = Path(output_md_path).parent
    audio_rel_path = os.path.relpath(p_original_audio_path, output_dir)
    srt_rel_path = os.path.relpath(output_dir / f"{p_original_audio_path.stem}.srt", output_dir)
    refined_rel_path = os.path.relpath(p_refined_text_path, output_dir)

    audio_link = f"[{p_original_audio_path.name}](./{audio_rel_path})"
    srt_link = f"[{Path(srt_rel_path).name}](./{srt_rel_path})"
    refined_text_link = f"[{Path(refined_rel_path).name}](./{refined_rel_path})"

    prompt = f"""
您是一位专业的会议助理。请根据下方提供的优化后会议文稿和文件元信息，生成一份详细、有条理的中文会议纪要。

**指令：**
1.  **提取元信息**:
    *   **会议主题**: 从文稿内容中提炼出核心议题作为纪要标题。
    *   **会议日期**: 尝试从文件名 `{p_original_audio_path.name}` 中推断日期（例如 '2023-10-26'）。如果无法推断，请使用今天的日期：{datetime.now().strftime('%Y-%m-%d')}。
    *   **参会人员**: 尝试从文稿中识别出参会者的姓名。如果无法明确识别，请注明“参会人员未明确”。

2.  **构建纪要结构 (Markdown格式)**: 纪要必须包含以下几个部分：
    *   `## 会议详情`: 包括推断出的会议主题、日期和参会人员。
    *   `## 纪要概述`: 一段话，简明扼要地总结会议的核心目的和主要成果。
    *   `## 主要讨论点`: 使用项目符号（bullet points）列出会议中讨论的主要议题。
    *   `## 行动项`: 使用编号列表（numbered list）明确记录需要执行的任务。如果可能，请指明负责人（例如：“1. 张三负责发送项目报告。”）。如果没有明确的行动项，请注明“无”。
    *   `## 关键决策`: 使用项目符号记录会议中达成的关键决定。如果没有，请注明“无”。

3.  **包含相关文件链接**: 在纪要末尾，添加 `## 相关文件` 部分，并附上以下链接：
    - 原始录音: {audio_link}
    - SRT字幕: {srt_link}
    - 精校文本: {refined_text_link}

**优化后的会议文稿:**
---
{refined_text}
---

请现在开始生成完整的中文Markdown格式会议纪要。
"""

    print("正在连接大语言模型以生成会议纪要...")
    client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
    response = client.chat.completions.create(
        model=config["model_name"],
        messages=[
            {"role": "system", "content": "你是一位专业的会议助理，请使用中文输出Markdown格式的会议纪要。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
    )
    summary_md = response.choices[0].message.content or ""

    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write(summary_md)
    print(f"会议纪要已保存至: {output_md_path}")
    return output_md_path
