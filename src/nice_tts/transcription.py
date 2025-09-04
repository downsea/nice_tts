import whisper
import os
from pathlib import Path

def transcribe_audio(audio_path: str, model_name: str = "base") -> str:
    """
    Transcribes an audio file using Whisper and saves the result as a TXT file.

    Args:
        audio_path (str): The path to the audio file.
        model_name (str): The name of the Whisper model to use (e.g., "tiny", "base", "small", "medium", "large").

    Returns:
        str: The path to the generated TXT file.
    """
    if not Path(audio_path).is_file():
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")

    print(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)

    print(f"Transcribing {audio_path}...")
    # Using verbose=False to prevent whisper from printing its own progress,
    # as we might want to have our own CLI progress indicators later.
    result = model.transcribe(audio_path, verbose=False)

    output_dir = Path(audio_path).parent
    audio_basename = Path(audio_path).name
    file_name_without_ext = Path(audio_path).stem

    # Get the TXT writer from whisper
    writer = whisper.utils.get_writer("txt", str(output_dir))

    # The writer creates a file named `audio_basename.txt`.
    # For example, for "example.wav", it creates "example.wav.txt".
    writer(result, audio_path)

    # We want the file to be named "example.txt", so we rename it.
    generated_txt_path = output_dir / f"{audio_basename}.txt"
    desired_txt_path = output_dir / f"{file_name_without_ext}.txt"

    os.rename(generated_txt_path, desired_txt_path)

    print(f"Transcription saved to {desired_txt_path}")

    return str(desired_txt_path)

if __name__ == '__main__':
    # This is an example of how to use the function.
    # To run this, you need to have an audio file named "example.wav" in the same directory.
    # For example, you can create a dummy wav file.
    # This block is for testing purposes and will not be executed when the module is imported.

    # Create a dummy wav file for testing if it doesn't exist.
    from scipy.io.wavfile import write
    import numpy as np

    dummy_audio_path = "example.wav"
    if not Path(dummy_audio_path).is_file():
        print("Creating a dummy audio file 'example.wav' for testing.")
        # Sample rate
        samplerate = 44100
        # Duration in seconds
        duration = 5
        # Frequency of the sine wave
        frequency = 440.0
        # Generate a sine wave
        t = np.linspace(0., duration, int(samplerate * duration), endpoint=False)
        amplitude = np.iinfo(np.int16).max * 0.5
        data = amplitude * np.sin(2. * np.pi * frequency * t)
        # Write to a WAV file
        write(dummy_audio_path, samplerate, data.astype(np.int16))
        print(f"'{dummy_audio_path}' created.")

    try:
        # Transcribe the dummy audio file
        txt_file_path = transcribe_audio(dummy_audio_path, model_name="tiny")
        print(f"Successfully transcribed '{dummy_audio_path}' to '{txt_file_path}'")

        # Print the content of the TXT file
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            print("\n--- TXT File Content ---")
            print(f.read())
            print("------------------------")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have ffmpeg installed and accessible in your system's PATH.")
        print("You can install it using: sudo apt-get install ffmpeg (on Debian/Ubuntu)")
