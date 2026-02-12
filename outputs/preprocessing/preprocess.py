from code.biological_constraints import apply_ip, create_tonotopic_mapping

import numpy as np
import librosa
import os
import soundfile as sf

folder_path = "../../data"
sr = 8000

def preprocess(folder_path, sr):
    for root_dir, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(".wav"):
                file_path = os.path.join(root_dir, filename)

                audio, orig_sr = librosa.load(file_path, sr=None)

                # PREPROCESS
                # Resample audio
                if orig_sr != sr:
                    audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)

                audio = rms_normalize(audio)

                rel_path = os.path.relpath(root_dir, folder_path)
                output_dir = os.path.join("../../preprocessed_data", rel_path)
                os.makedirs(output_dir, exist_ok=True)

                output_path = os.path.join(output_dir, filename)

                # Save
                sf.write(output_path, audio, sr)
                print(output_path)

def rms_normalize(audio):
    rms = np.sqrt(np.mean(audio**2))
    audio = audio / rms

    return audio

if __name__ == "__main__":
    preprocess(folder_path, sr)
