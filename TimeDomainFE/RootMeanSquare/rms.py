import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def load_signal(path):
    signal, sr = librosa.load(path)
    return signal, sr

def calculate_rms(signal, frame_size,hop_length):
    rms_signal=[]
    for frame in range(0,len(signal),hop_length):
        current_Frame_rms = np.sqrt(sum(signal[frame:frame+frame_size]**2) / frame_size)
        rms_signal.append(current_Frame_rms)
    return np.array(rms_signal)

def plot_rms_signal(rms_signal,original_signal,plotName,frame_length,hop_length,plot_title):
    frames = range(len(rms_signal))
    t = librosa.frames_to_time(frames,hop_length=hop_length)
    plt.figure(figsize=(25,12))
    librosa.display.waveplot(original_signal)
    plt.title(plot_title)
    plt.plot(t,rms_signal, color = 'red')
    # plt.show()
    plt.savefig(plotName+"png")

redhotPath = "../../audios/redhot.wav"
debussyPath = "../../audios/debussy.wav"
noisePath = "../../audios/noise.wav"

redhot, sr = load_signal(redhotPath)
debussy, _ = load_signal(debussyPath)
noise, _ = load_signal(noisePath)

redhot_rms = calculate_rms(signal=redhot,frame_size=1024,hop_length=512)
plot_rms_signal(rms_signal=redhot_rms,original_signal=redhot,plotName="RMS Redhot",plot_title="RMS Redhot display",frame_length=1024,hop_length=512)

debussy_rms = calculate_rms(signal=debussy,frame_size=1024,hop_length=512)
plot_rms_signal(rms_signal=debussy_rms,original_signal=debussy,plotName="Debussy Redhot",plot_title="Debussy RMS display",frame_length=1024,hop_length=512)

noise_rms = calculate_rms(signal=noise,frame_size=1024,hop_length=512)
plot_rms_signal(rms_signal=noise_rms,original_signal=noise,plotName="NOISE Redhot",plot_title="Noise RMS display",frame_length=1024,hop_length=512)
