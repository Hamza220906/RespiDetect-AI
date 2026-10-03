import os

audio_files = [f for f in os.listdir("data/audio_files") if f.endswith(".wav")]
test_audio = audio_files[0]
base_name = test_audio.replace(".wav", "")


label_file = os.path.join("data/audio_files", base_name + ".txt")

if os.path.exists(label_file):
    print(f"📄 Found label file: {base_name}.txt")
    
    
    with open(label_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
    print(f"🔍 Raw Label Content:\n{content}")
    
    
    parts = content.split('\t')
    if len(parts) >= 6:
        disease_class = parts[5] 
        print(f"\n✅ Disease Class Detected: '{disease_class}'")
        
        if disease_class.strip() == "" or disease_class.strip() == "-":
            print("🟢 This is likely a HEALTHY recording.")
        else:
            print(f"🔴 This is a DISEASED recording ({disease_class}).")
    else:
        print("⚠️ Could not parse standard ICBHI format. Raw content shown above.")
else:
    print(f"❌ Label file NOT found for {test_audio}")