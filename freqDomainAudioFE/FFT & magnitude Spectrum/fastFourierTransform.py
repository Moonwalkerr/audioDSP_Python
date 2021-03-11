import numpy as np
import librosa 
import librosa.display
import matplotlib.pyplot as plt

# voilin_file = "../audios/violin_c.wav"
sax_file = "../audios/sax.wav"
piano_file = "../audios/piano_c.wav"
noise_file = "../audios/noise.wav"


# voilin, sr = librosa.load(voilin_file)
sax , _ = librosa.load(sax_file)
piano, _ = librosa.load(piano_file)
noise, _ = librosa.load(noise_file)

voilin_ft = np.fft.fft(sax)
print(voilin_ft)