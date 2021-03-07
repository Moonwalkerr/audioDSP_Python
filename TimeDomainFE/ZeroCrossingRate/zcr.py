import librosa
import librosa.display
import matplotlib.pyplot as plt

debussyPath = "../../audios/debussy.wav"
redhotPath = "../../audios/redhot.wav"
noisePath = "../../audios/noise.wav"

def loadSignal(signalPath):
    signal, sr = librosa.load(signalPath)
    return signal, sr
def calculate_ZeroCrossingRate(signal, frame_size, hop_length):
    zcr_signal = librosa.feature.zero_crossing_rate(signal,frame_length=frame_size,hop_length=hop_length)[0]
    return zcr_signal
def plot_ZCR(zcr_signal, frame_length, hop_length,plot_title,figname):
    frames = range(len(zcr_signal))
    t = librosa.frames_to_time(frames,hop_length=hop_length)
    plt.figure(figsize=(25,12))
    plt.plot(t,zcr_signal*frame_length, color='r')
    plt.ylim((0, 1))
    plt.title(plot_title)
    plt.savefig(figname)

noise, sr = loadSignal(noisePath)
debussy, _ = loadSignal(debussyPath)
redhot, _ = loadSignal(redhotPath)

debussy_zcr = calculate_ZeroCrossingRate(signal=debussy,frame_size=2048,hop_length=512)
plot_ZCR(zcr_signal=debussy_zcr,frame_length=2048,hop_length=512,plot_title="Zero Crossing Rate Display of Debussy", figname="Debussy_ZCR")


redhot_zcr = calculate_ZeroCrossingRate(signal=redhot,frame_size=2048,hop_length=512)
plot_ZCR(zcr_signal=redhot_zcr,frame_length=2048,hop_length=512,plot_title="Zero Crossing Rate Display of Redhot", figname="Redhot_ZCR")


noise_zcr = calculate_ZeroCrossingRate(signal=noise,frame_size=2048,hop_length=512)
plot_ZCR(zcr_signal=noise_zcr,frame_length=2048,hop_length=512,plot_title="Zero Crossing Rate Display of Noise", figname="Noise_ZCR")

print("Done !")