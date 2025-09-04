import whisper
import os
from pathlib import Path

def transcribe_audio(
    audio_path: str, output_srt_path: str, model_name: str = "base", language: str = "en"
) -> str:
    """
    Transcribes an audio file and saves the result to a specified SRT file path.

    Args:
        audio_path (str): The path to the input audio file.
        output_srt_path (str): The full path where the output .srt file will be saved.
        model_name (str): The name of the Whisper model to use.
        language (str): The language of the audio. Use 'zh' for Chinese, 'en' for English.

    Returns:
        str: The path to the generated SRT file.
    """
    if not Path(audio_path).is_file():
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")

    print(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)

    print(f"Transcribing {audio_path} (Language: {language})...")
    result = model.transcribe(audio_path, language=language, verbose=False)

    # Manually generate SRT content for full control over the output path.
    srt_content = ""
    for i, segment in enumerate(result['segments']):
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text']

        start_h, rem = divmod(start_time, 3600)
        start_m, rem = divmod(rem, 60)
        start_s, start_ms = divmod(rem, 1)

        end_h, rem = divmod(end_time, 3600)
        end_m, rem = divmod(rem, 60)
        end_s, end_ms = divmod(rem, 1)

        srt_content += f"{i + 1}\n"
        srt_content += f"{int(start_h):02}:{int(start_m):02}:{int(start_s):02},{int(start_ms*1000):03} --> {int(end_h):02}:{int(end_m):02}:{int(end_s):02},{int(end_ms*1000):03}\n"
        srt_content += f"{text.strip()}\n\n"

    with open(output_srt_path, 'w', encoding='utf-8') as f:
        f.write(srt_content)

    print(f"Transcription saved to {output_srt_path}")

    return output_srt_path

if __name__ == '__main__':
    # This block is for testing purposes.
    from scipy.io.wavfile import write
    import numpy as np

    test_dir = Path("test_transcription_output")
    test_dir.mkdir(exist_ok=True)
    dummy_audio = test_dir / "example_test.wav"
    dummy_srt = test_dir / "example_test.srt"

    if not Path(dummy_audio).is_file():
        print(f"Creating a dummy audio file '{dummy_audio}' for testing.")
        samplerate = 44100
        duration = 3
        frequency = 440.0
        t = np.linspace(0., duration, int(samplerate * duration), endpoint=False)
        amplitude = np.iinfo(np.int16).max * 0.5
        data = amplitude * np.sin(2. * np.pi * frequency * t)
        write(dummy_audio, samplerate, data.astype(np.int16))

    try:
        print("--- Testing Transcription (English) ---")
        transcribe_audio(
            audio_path=str(dummy_audio),
            output_srt_path=str(dummy_srt),
            model_name="tiny",
            language="en"
        )
        with open(dummy_srt, 'r', encoding='utf-8') as f:
            print("\n--- SRT File Content ---")
            print(f.read())
            print("------------------------")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure `ffmpeg` and `scipy` are installed.")
