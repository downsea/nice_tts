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
    # Load global .env from home directory
    global_dotenv_path = Path.home() / ".env"
    if global_dotenv_path.is_file():
        load_dotenv(dotenv_path=global_dotenv_path)

    # Load local .env, overriding global settings
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
    Refines a transcript from an SRT file and saves it to a specified path.
    """
    p_srt_path = Path(srt_path)
    if not p_srt_path.is_file():
        raise FileNotFoundError(f"SRT file not found at: {srt_path}")

    config = load_llm_config()
    with open(p_srt_path, 'r', encoding='utf-8') as f:
        srt_content = f.read()

    transcript_text = _parse_srt_to_text(srt_content)
    if not transcript_text:
        print("Warning: Could not extract any text from the SRT file.")
        Path(output_fine_path).touch()
        return output_fine_path

    prompt = f"""
You are a highly skilled editor. Your task is to refine the following raw, auto-generated transcript of a meeting or recording. The goal is to produce a clean, readable, and well-formatted text that accurately represents the original conversation. Please perform the following actions:
1.  **Correct Punctuation and Capitalization**: Ensure proper sentence structure.
2.  **Fix Spelling and Grammatical Errors**.
3.  **Remove Filler Words and Repetitions** (e.g., "um", "uh", "like").
4.  **Merge Broken Sentences** into coherent, complete sentences.
5.  **Structure into Paragraphs**.
Do not add any headers, titles, or speaker labels. The output should be only the refined text itself.

Here is the raw transcript:
---
{transcript_text}
---
"""

    print("Connecting to LLM for transcript refinement...")
    client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
    response = client.chat.completions.create(
        model=config["model_name"],
        messages=[
            {"role": "system", "content": "You are an expert editor tasked with refining raw transcripts."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    refined_text = response.choices[0].message.content or ""

    with open(output_fine_path, 'w', encoding='utf-8') as f:
        f.write(refined_text)
    print(f"Refined transcript saved to {output_fine_path}")
    return output_fine_path

def summarize_transcript(
    refined_text_path: str, original_audio_path: str, output_md_path: str
) -> str:
    """
    Generates a meeting summary and saves it to a specified path.
    """
    p_refined_text_path = Path(refined_text_path)
    p_original_audio_path = Path(original_audio_path)
    if not p_refined_text_path.is_file():
        raise FileNotFoundError(f"Refined transcript file not found at: {refined_text_path}")

    config = load_llm_config()
    with open(p_refined_text_path, 'r', encoding='utf-8') as f:
        refined_text = f.read()

    if not refined_text.strip():
        print("Warning: Refined transcript is empty. Cannot generate summary.")
        Path(output_md_path).touch()
        return output_md_path

    srt_path = (Path(output_md_path).parent / f"{p_original_audio_path.stem}.srt").relative_to(Path(output_md_path).parent)
    audio_link = f"[{p_original_audio_path.name}](./{p_original_audio_path.name})"
    srt_link = f"[{srt_path.name}](./{srt_path.name})"
    refined_text_link = f"[{p_refined_text_path.name}](./{p_refined_text_path.name})"

    prompt = f"""
You are a professional assistant responsible for creating concise and accurate meeting summaries in Markdown format. Based on the provided transcript and file metadata, please perform the following:
1.  **Infer Metadata**: Infer the meeting Topic, Date (from filename `{p_original_audio_path.name}` or use today's date: {datetime.now().strftime('%Y-%m-%d')}), and Participants.
2.  **Structure the Summary**: Create sections for `## Meeting Details`, `## Overview`, `## Key Discussion Points`, `## Action Items`, and `## Decisions Made`.
3.  **Include File Links**: At the end, add a `## Related Files` section with these links:
    - Original Audio: {audio_link}
    - SRT Transcript: {srt_link}
    - Refined Transcript: {refined_text_link}

Input Transcript:
---
{refined_text}
---
"""

    print("Connecting to LLM for summarization...")
    client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
    response = client.chat.completions.create(
        model=config["model_name"],
        messages=[
            {"role": "system", "content": "You are a professional assistant that writes meeting summaries."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
    )
    summary_md = response.choices[0].message.content or ""

    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write(summary_md)
    print(f"Meeting summary saved to {output_md_path}")
    return output_md_path

if __name__ == '__main__':
    print("--- Setting up for LLM testing ---")
    if not Path(".env").is_file():
        print("Creating a dummy .env file. Please fill it with your credentials.")
        with open(".env", "w") as f:
            f.write('OPENAI_API_KEY="your_api_key_here"\n')
            f.write('OPENAI_API_BASE="https://api.openai.com/v1"\n')
            f.write('OPENAI_MODEL_NAME="gpt-3.5-turbo"\n')

    test_dir = Path("test_llm_output")
    test_dir.mkdir(exist_ok=True)
    dummy_srt_path = test_dir / "example_llm_test.srt"
    dummy_fine_path = test_dir / "example_llm_test.fine.txt"
    dummy_md_path = test_dir / "example_llm_test.md"
    dummy_audio_path = "example_llm_test.wav"

    if not Path(dummy_srt_path).is_file():
        with open(dummy_srt_path, "w", encoding="utf-8") as f:
            f.write("1\n00:00:01,123 --> 00:00:03,456\nhello world\n\n")

    try:
        print("\n--- Testing Refinement ---")
        refine_transcript(str(dummy_srt_path), str(dummy_fine_path))
        print(f"Refinement test called. Check for '{dummy_fine_path}'.")

        if Path(dummy_fine_path).is_file():
             print("\n--- Testing Summarization ---")
             summarize_transcript(str(dummy_fine_path), dummy_audio_path, str(dummy_md_path))
             print(f"Summarization test called. Check for '{dummy_md_path}'.")

    except Exception as e:
        print(f"\nAn error occurred during LLM testing: {e}")
