import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm


os.makedirs("outputs/spectrograms", exist_ok=True)

data_dir = "data/audio_files"
audio_files = [f for f in os.listdir(data_dir) if f.endswith(".wav")]

print(f" Generating spectrograms for {len(audio_files)} files...")

for audio_file in tqdm(audio_files):
    try:
        
        y, sr = librosa.load(os.path.join(data_dir, audio_file), sr=44100)
        
        
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        
        save_path = os.path.join("outputs/spectrograms", audio_file.replace(".wav", ".png"))
        plt.figure(figsize=(4, 3))
        plt.imshow(mel_spec_db, aspect='auto', cmap='magma')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()
        
    except Exception as e:
        print(f"\n⚠️ Failed to process {audio_file}: {str(e)}")

print("\n✅ All spectrograms saved to outputs/spectrograms/")