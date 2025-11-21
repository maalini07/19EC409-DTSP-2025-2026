# EXP 1(C) : Analysis of audio signal for noise removal

# AIM: 

# To analyse an audio signal and remove noise

# APPARATUS REQUIRED:  
PC installed with SCILAB. 

# PROGRAM: 
```
# ==============================
# AUDIO NOISE REMOVAL & SEPARATION
# ==============================

# Step 1: Install packages
!pip install -q librosa noisereduce soundfile

# Step 2: Upload clean and noise recordings
from google.colab import files
print("Upload clean/normal audio (speech/music)")
uploaded = files.upload()
clean_file = next(iter(uploaded.keys()))

print("Upload noise-only audio (background)")
uploaded = files.upload()
noise_file = next(iter(uploaded.keys()))

# Step 3: Load audios
import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import noisereduce as nr

clean, sr_c = librosa.load(clean_file, sr=None, mono=True)
noise, sr_n = librosa.load(noise_file, sr=None, mono=True)

# 🔧 Resample noise if sample rates differ
if sr_c != sr_n:
    print(f"Resampling noise from {sr_n} Hz → {sr_c} Hz")
    noise = librosa.resample(noise, orig_sr=sr_n, target_sr=sr_c)
    sr_n = sr_c

sr = sr_c
print(f"Clean audio SR = {sr_c}, Noise audio SR = {sr_n}")
print(f"Clean length = {len(clean)/sr:.2f} sec, Noise length = {len(noise)/sr:.2f} sec")

# Step 4: Make lengths equal (pad or cut noise)
if len(noise) < len(clean):
    reps = int(np.ceil(len(clean)/len(noise)))
    noise = np.tile(noise, reps)[:len(clean)]
else:
    noise = noise[:len(clean)]

# Step 5: Create noisy mixture
noisy = clean + noise * 0.5   # adjust noise scaling factor
print("Generated noisy signal.")

# Step 6: Play audio
print("\n--- Original Clean Audio ---")
display(Audio(clean, rate=sr))

print("\n--- Noise Sample ---")
display(Audio(noise, rate=sr))

print("\n--- Noisy (Merged) Audio ---")
display(Audio(noisy, rate=sr))

# Step 7: Frequency analysis (FFT spectra)
def plot_spectrum(signal, sr, title):
    n_fft = 2**14
    Y = np.fft.rfft(signal, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    magnitude = np.abs(Y)
    plt.figure(figsize=(12,4))
    plt.semilogy(freqs, magnitude+1e-12)
    plt.xlim(0, sr/2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (log)")
    plt.title(title)
    plt.grid(True)
    plt.show()

plot_spectrum(clean, sr, "Spectrum of Clean Audio")
plot_spectrum(noise, sr, "Spectrum of Noise")
plot_spectrum(noisy, sr, "Spectrum of Noisy Audio")

# Step 8: Noise reduction (spectral subtraction)
reduced = nr.reduce_noise(y=noisy, y_noise=noise, sr=sr)

# Step 9: Separate estimated noise = noisy - reduced
estimated_noise = noisy - reduced

print("\n--- Denoised / Cleaned Audio ---")
display(Audio(reduced, rate=sr))

print("\n--- Extracted Noise Component ---")
display(Audio(clean, rate=sr))

# Step 10: Compare spectrograms
def plot_spec(signal, sr, title):
    D = librosa.stft(signal, n_fft=1024, hop_length=512)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    plt.figure(figsize=(12,5))
    librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis="time", y_axis="hz")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.ylim(0, sr/2)
    plt.show()

plot_spec(noisy, sr, "Spectrogram of Noisy Audio")
plot_spec(reduced, sr, "Spectrogram of Denoised Audio")
plot_spec(estimated_noise, sr, "Spectrogram of Extracted Noise")
```
## ORIGINAL CLEAN AUDIO:
download.wav

## NOISE SAMPLE:
download (1).wav

## NOISE MERGED AUDIO:
download (2).wav

## EXTRACTED NOISE REMOVAL:
download (3).wav
## OUTPUT
<img width="1259" height="359" alt="Screenshot 2025-11-16 234601" src="https://github.com/user-attachments/assets/f2209901-c674-4851-9ca5-db5818e1dcc6" />

<img width="1269" height="418" alt="Screenshot 2025-11-16 234620" src="https://github.com/user-attachments/assets/29d3c48e-3015-432c-8405-1814182358a8" />

<img width="1271" height="362" alt="Screenshot 2025-11-16 234643" src="https://github.com/user-attachments/assets/1771c77c-f71f-4022-946c-6a830c481310" />

<img width="1196" height="1025" alt="image" src="https://github.com/user-attachments/assets/06f285e8-3459-4c65-b1ec-67d0248392fa" />

<img width="1266" height="555" alt="image" src="https://github.com/user-attachments/assets/cbf2260f-56d8-4e55-8bd4-cffa778a43d9" />

# RESULT: 
Thus,the Analysis of audio signal for noise removal is verified
