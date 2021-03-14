import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

violin_file = "../../audios/violin_c.wav"
sax_file = "../../audios/sax.wav"
piano_file = "../../audios/piano_c.wav"
noise_file = "../../audios/noise.wav"

violin, sr = librosa.load(violin_file)
sax , _ = librosa.load(sax_file)
piano, _ = librosa.load(piano_file)
noise, _ = librosa.load(noise_file)

# extracting short time fourier transforms of the signals
FRAME_SIZE = 2048
HOP_LENGTH = 512

violin_stft = librosa.stft(violin, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)
# print(type(violin_stft[0][0]))  --> numpy.complex64

# we have extracted the stft from the signal
# Now we need to visualize it using spectorgram:- for that we need to get rid of complex nos 
# we have to take sqaured magnitude of the stft output
Y_scale = np.abs(violin_stft)**2
# print(type(Y_scale[0][0]))  #--> numpy.float32

# currently our Y_scale is in linear format
# we need to convert into decibles / human perceivable format using logarithmic function
# thus transforming the intensities / amplitudes
Y_scale_log = librosa.power_to_db(Y_scale)


def plot_spectogram(Y, sr , hop_length, y_axis= 'log',title="Spectogram",savefigname="Signal"):
    plt.figure(figsize=(10,10))
    librosa.display.specshow(Y, sr=sr, hop_length=hop_length, x_axis='time',y_axis=y_axis)
    plt.colorbar(format='%+2.f')
    plt.title("Spectogram of {}".format(title))
    # plt.show()
    plt.savefig("SpecOf{}".format(savefigname))

# plot_spectogram(Y_scale_log,sr,HOP_LENGTH,title="Spectorgam of Violin Signal",savefigname="Violin")


def extract_plot_spectogram(Signal, sr ,frame_size, hop_length, title="Spectogram",savefigname="Signal"):
    y_scale = librosa.stft(Signal,n_fft=frame_size,hop_length=hop_length)
    y_scale_log = librosa.power_to_db(y_scale)
    plt.figure(figsize=(20,10))
    librosa.display.specshow(y_scale_log,sr=sr,hop_length=hop_length,x_axis='time',y_axis='log')
    plt.colorbar()
    plt.title("Spectogram of {}".format(title))
    plt.savefig("SpecOf{}".format(savefigname))

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
    extract_plot_spectogram(sig["sig"],sr=sr,frame_size=FRAME_SIZE,hop_length=HOP_LENGTH,title="Spectogram of {}".format(sig["sigName"]),savefigname="SpecOf{}".format(sig["sigName"]))
    
print("Done !")
