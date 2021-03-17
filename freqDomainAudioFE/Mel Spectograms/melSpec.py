import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

violin_file = "../../audios/violin_c.wav"
sax_file = "../../audios/sax.wav"
piano_file = "../../audios/piano_c.wav"
noise_file = "../../audios/noise.wav"
scale_file = "../../audios/scale.wav"


violin, sr = librosa.load(violin_file)
sax , _ = librosa.load(sax_file)
piano, _ = librosa.load(piano_file)
noise, _ = librosa.load(noise_file)
scale, _ = librosa.load(scale_file)
# extracting short time fourier transforms of the signals
FRAME_SIZE = 2048
HOP_LENGTH = 512


filter_banks = librosa.filters.mel(sr,n_fft=FRAME_SIZE,n_mels=10)


# visualizing the Mel Filter banks 
# plt.figure(figsize=(20,10))
# librosa.display.specshow(filter_banks,sr=sr,x_axis="linear")
# plt.colorbar()
# plt.show()


# extracting the mel spectogram from the signal
# this feature of the librosa first extracts the vanilla spectogram
# then generates the Mel Spectogram all under the hood in one go
scale_mel_spec = librosa.feature.melspectrogram(scale,hop_length=HOP_LENGTH,n_fft=FRAME_SIZE,sr=sr,n_mels=10)
# the lower the n_mels be , the more understandable and visible the freq bins would be


# power --> db
log_mel_spec = librosa.power_to_db(scale_mel_spec)


# visualizing the Mel spectogram
# plt.figure(figsize=(20,10))
# librosa.display.specshow(log_mel_spec,x_axis='time',y_axis='mel')
# plt.colorbar()
# # plt.show()
# plt.savefig("Spectogram of Scale Signal")


def extract_plot_MelSpec(signal, frame_size,hop_length,sr,title="MelSpectogram",savefigname="MelSpec",n_mels=10):
    mel_spec = librosa.feature.melspectrogram(signal, n_fft=frame_size,hop_length=hop_length,sr=sr,n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec)
    plt.figure(figsize=(20,10))
    librosa.display.specshow(mel_spec_db,x_axis='time',y_axis='mel')
    plt.colorbar()
    plt.title(title)
    plt.savefig(savefigname)
signal1 = {
    "sig": sax,
    "sigName": "sax"
    }

signal2 = {
    "sig": violin,
    "sigName": "violin"
    }

signal3 = {
    "sig": piano,
    "sigName": "piano"
    }

signal4 = {
    "sig": violin,
    "sigName": "violin"
    }

signals = [signal1,signal2,signal3,signal4]
for sig in signals:
    extract_plot_MelSpec(sig["sig"],sr=sr,frame_size=FRAME_SIZE,hop_length=HOP_LENGTH,title="Mel Spectogram of {}".format(sig["sigName"]),savefigname="MelSpec{}".format(sig["sigName"]))
    
print("Done !")
