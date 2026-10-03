import os
from collections import Counter

data_dir = "data/audio_files"
audio_files = [f for f in os.listdir(data_dir) if f.endswith(".wav")]

labels = []
missing_labels = []

for audio in audio_files:
    base_name = audio.replace(".wav", "")
    txt_path = os.path.join(data_dir, base_name + ".txt")
    
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        
        file_label = "Unknown"
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                
                code = parts[3].strip()
                if code != "0" and code != "":
                    file_label = "Diseased"
                    break
                else:
                    file_label = "Healthy"
        
        labels.append(file_label)
    else:
        missing_labels.append(audio)


label_counts = Counter(labels)
print(f"📊 Total Audio Files: {len(audio_files)}")
print(f"✅ Healthy Recordings: {label_counts.get('Healthy', 0)}")
print(f"🔴 Diseased Recordings: {label_counts.get('Diseased', 0)}")
print(f"️ Missing Labels: {len(missing_labels)}")

if missing_labels:
    print("\nFiles without labels:", missing_labels[:5], "...")