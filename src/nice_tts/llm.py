import os
import re
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

def load_llm_config():
    """
    Loads LLM configuration from a .env file.
    Looks for OPENAI_API_KEY, OPENAI_API_BASE, and OPENAI_MODEL_NAME.
    """
    load_dotenv()
    config = {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_API_BASE"),
        "model_name": os.getenv("OPENAI_MODEL_NAME"),
    }
    if not all(config.values()):
        raise ValueError(
            "Missing one or more required environment variables: "
            "OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_MODEL_NAME. "
            "Please create a .env file with these values."
        )
    return config

def _parse_srt_to_text(srt_content: str) -> str:
    """
    Parses SRT content to extract plain text, ignoring timestamps and sequence numbers.
    """
    lines = srt_content.strip().split('\n')
    text_parts = []
    for line in lines:
        line = line.strip()
        if line and not line.isdigit() and '-->' not in line:
            text_parts.append(line)
    return " ".join(text_parts)

def refine_transcript(srt_path: str) -> str:
    """
    Refines a transcript from an SRT file using an LLM.

    Args:
        srt_path (str): The path to the SRT transcript file.

    Returns:
        str: The path to the refined text file.
    """
    p_srt_path = Path(srt_path)
    if not p_srt_path.is_file():
        raise FileNotFoundError(f"SRT file not found at: {srt_path}")

    # Load LLM configuration
    try:
        config = load_llm_config()
    except ValueError as e:
        print(f"Error: {e}")
        raise

    # Read and parse SRT content
    with open(p_srt_path, 'r', encoding='utf-8') as f:
        srt_content = f.read()

    transcript_text = _parse_srt_to_text(srt_content)
    if not transcript_text:
        print("Warning: Could not extract any text from the SRT file.")
        # Create an empty refined file and return
        output_path = p_srt_path.with_suffix('.fine.txt')
        output_path.touch()
        return str(output_path)

    # Prepare the prompt for the LLM
    prompt = f"""
You are a highly skilled editor. Your task is to refine the following raw, auto-generated transcript of a meeting or recording.
The goal is to produce a clean, readable, and well-formatted text that accurately represents the original conversation.

Please perform the following actions:
1.  **Correct Punctuation and Capitalization**: Ensure proper sentence structure, using commas, periods, and capitalization correctly.
2.  **Fix Spelling and Grammatical Errors**: Correct any spelling mistakes or grammatical issues.
3.  **Remove Filler Words and Repetitions**: Eliminate conversational fillers (e.g., "um", "uh", "like", "you know") and unnecessary repetitions, unless they are essential for context.
4.  **Merge Broken Sentences**: Combine fragmented sentences that were split during transcription into coherent, complete sentences.
5.  **Structure into Paragraphs**: Group related sentences into logical paragraphs to improve readability. Do not label speakers. The output should be a continuous text.
6.  **Preserve Meaning**: It is crucial that you do not change the original meaning of the conversation. The refined text must remain faithful to the source.

Do not add any headers, titles, or speaker labels. The output should be only the refined text itself.

Here is the raw transcript:
---
{transcript_text}
---
"""

    # Call the LLM
    print("Connecting to LLM for transcript refinement...")
    try:
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
    except Exception as e:
        print(f"An error occurred while communicating with the LLM: {e}")
        raise

    # Save the refined text
    output_path = p_srt_path.with_suffix('.fine.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(refined_text)

    print(f"Refined transcript saved to {output_path}")
    return str(output_path)

from datetime import datetime


def summarize_transcript(refined_text_path: str, original_audio_path: str) -> str:
    """
    Generates a meeting summary from a refined transcript using an LLM.

    Args:
        refined_text_path (str): The path to the refined transcript file (.fine.txt).
        original_audio_path (str): The path to the original audio file, used for metadata.

    Returns:
        str: The path to the generated Markdown summary file.
    """
    p_refined_text_path = Path(refined_text_path)
    p_original_audio_path = Path(original_audio_path)

    if not p_refined_text_path.is_file():
        raise FileNotFoundError(f"Refined transcript file not found at: {refined_text_path}")

    # Load LLM configuration
    config = load_llm_config()

    # Read refined transcript
    with open(p_refined_text_path, 'r', encoding='utf-8') as f:
        refined_text = f.read()

    if not refined_text.strip():
        print("Warning: Refined transcript is empty. Cannot generate summary.")
        output_path = p_original_audio_path.with_suffix('.md')
        output_path.touch()
        return str(output_path)

    # Prepare file links for the summary
    srt_path = p_original_audio_path.with_suffix('.srt')

    # Using relative paths for portability in the final markdown
    audio_link = f"[{p_original_audio_path.name}](./{p_original_audio_path.name})"
    srt_link = f"[{srt_path.name}](./{srt_path.name})"
    refined_text_link = f"[{p_refined_text_path.name}](./{p_refined_text_path.name})"

    # Prepare the prompt for the LLM
    prompt = f"""
You are a professional assistant responsible for creating concise and accurate meeting summaries.
Based on the provided transcript and file metadata, generate a detailed meeting summary in Markdown format.

**Instructions:**
1.  **Infer Metadata**:
    *   **Topic**: Infer the main topic or title of the meeting from the transcript.
    *   **Date**: Infer the date from the filename `{p_original_audio_path.name}` if possible (e.g., from '2023-10-26'). If not, use today's date: {datetime.now().strftime('%Y-%m-%d')}.
    *   **Participants**: Infer the names of the participants from the conversation if possible. If not, state that participants are not clearly identifiable.

2.  **Structure the Summary**: The summary must have the following sections:
    *   `## Meeting Details`: Include the inferred Topic, Date, and Participants.
    *   `## Overview`: A brief, one-paragraph summary of the meeting's purpose and key outcomes.
    *   `## Key Discussion Points`: A bulleted list of the main topics that were discussed.
    *   `## Action Items`: A numbered list of tasks or action items. For each item, specify who is responsible if mentioned (e.g., "John to send the report."). If no action items, state "None."
    *   `## Decisions Made`: A bulleted list of key decisions reached during the meeting. If no decisions were made, state "None."

3.  **Include File Links**: At the end of the summary, include a section `## Related Files` with links to the original audio, SRT transcript, and refined text.

**Input Transcript:**
---
{refined_text}
---

**File Links to Include:**
- Original Audio: {audio_link}
- SRT Transcript: {srt_link}
- Refined Transcript: {refined_text_link}

Please generate the complete Markdown summary now.
"""

    # Call the LLM
    print("Connecting to LLM for summarization...")
    try:
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
    except Exception as e:
        print(f"An error occurred while communicating with the LLM: {e}")
        raise

    # Save the summary
    output_path = p_original_audio_path.with_suffix('.md')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(summary_md)

    print(f"Meeting summary saved to {output_path}")
    return str(output_path)


if __name__ == '__main__':
    # This block is for testing the LLM logic.
    # It requires a .env file with valid LLM credentials.

    # --- Shared Setup for Testing ---
    print("--- Setting up for testing ---")
    # Create a dummy .env file for testing if it doesn't exist
    if not Path(".env").is_file():
        print("Creating a dummy .env file. Please fill it with your credentials.")
        with open(".env", "w") as f:
            f.write('OPENAI_API_KEY="your_api_key_here"\n')
            f.write('OPENAI_API_BASE="https://api.openai.com/v1"\n')
            f.write('OPENAI_MODEL_NAME="gpt-3.5-turbo"\n')
        print("Please edit the .env file with your actual LLM provider details.")

    # --- Test for Refinement ---
    print("\n--- Testing Transcript Refinement ---")
    dummy_srt_path = "example.srt"
    if not Path(dummy_srt_path).is_file():
        print(f"Creating a dummy SRT file '{dummy_srt_path}' for testing.")
        with open(dummy_srt_path, "w", encoding="utf-8") as f:
            f.write("1\n00:00:01,123 --> 00:00:03,456\nokay so um i think we can start the meeting now\n\n")

    print("Attempting to refine the dummy transcript...")
    print("NOTE: This will fail if you have not configured a valid .env file.")

    try:
        refined_file = refine_transcript(dummy_srt_path)
        print(f"Successfully refined transcript. Output at: {refined_file}")
    except Exception as e:
        print(f"Refinement process failed as expected (or unexpectedly): {e}")

    # --- Test for Summarization ---
    print("\n--- Testing Transcript Summarization ---")
    dummy_audio_path = "meeting_2023-10-26_project-kickoff.wav"
    dummy_refined_path = "meeting_2023-10-26_project-kickoff.fine.txt"
    if not Path(dummy_refined_path).is_file():
         print(f"Creating a dummy refined text file '{dummy_refined_path}' for testing.")
         with open(dummy_refined_path, "w", encoding="utf-8") as f:
            f.write("The team discussed the project kickoff for the new mobile app. John will handle the frontend, and Sarah will manage the backend services. A decision was made to use React Native. The next meeting is scheduled for next Tuesday.")

    print(f"Attempting to summarize the dummy transcript '{dummy_refined_path}'...")
    print("NOTE: This will also fail if you have not configured a valid .env file.")

    try:
        summary_file = summarize_transcript(dummy_refined_path, dummy_audio_path)
        print(f"Successfully generated summary. Output at: {summary_file}")
        with open(summary_file, 'r', encoding='utf-8') as f:
            print("\n--- Generated Summary ---")
            print(f.read())
            print("-------------------------")
    except Exception as e:
        print(f"Summarization process failed as expected (or unexpectedly): {e}")
