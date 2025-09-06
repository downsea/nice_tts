import whisper
import os
from pathlib import Path

def transcribe_audio(
    audio_path: str, output_txt_path: str, model_name: str = "base", language: str = "en"
) -> str:
    """
    Transcribes an audio file and saves the result to a specified TXT file path.

    Args:
        audio_path (str): The path to the input audio file.
        output_txt_path (str): The full path where the output .txt file will be saved.
        model_name (str): The name of the Whisper model to use.
        language (str): The language of the audio. Use 'zh' for Chinese, 'en' for English.

    Returns:
        str: The path to the generated TXT file.
    """
    if not Path(audio_path).is_file():
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")

    print(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)

    print(f"Transcribing {audio_path} (Language: {language})...")
    result = model.transcribe(audio_path, language=language, verbose=False)

    transcribed_text = result['text'].strip()

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(transcribed_text)

    print(f"Transcription saved to {output_txt_path}")

    return output_txt_path

if __name__ == '__main__':
    # This block is for testing purposes.
    from scipy.io.wavfile import write
    import numpy as np

    test_dir = Path("test_transcription_output")
    test_dir.mkdir(exist_ok=True)
    dummy_audio = test_dir / "example_test.wav"
    dummy_txt = test_dir / "example_test.txt"

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
            output_txt_path=str(dummy_txt),
            model_name="tiny",
            language="en"
        )
        with open(dummy_txt, 'r', encoding='utf-8') as f:
            print("\n--- TXT File Content ---")
            print(f.read())
            print("------------------------")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure `ffmpeg` and `scipy` are installed.")
