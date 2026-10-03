import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os


audio_files = [f for f in os.listdir("data/audio_files") if f.endswith(".wav")]
if not audio_files:
    print(" No .wav files found!")
else:
    test_file = os.path.join("data/audio_files", audio_files[0])
    print(f"🎵 Testing file: {test_file}")

    
    y, sr = librosa.load(test_file, sr=44100)
    print(f"⏱️ Duration: {len(y)/sr:.2f}s | Sample Rate: {sr} Hz")

  
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram: {os.path.basename(test_file)}')
    plt.tight_layout()
    
    os.makedirs("outputs", exist_ok=True)
    save_path = "outputs/sample_spectrogram.png"
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"✅ Saved to {save_path}")