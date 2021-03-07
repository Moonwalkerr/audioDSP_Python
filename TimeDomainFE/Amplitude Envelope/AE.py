import  librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

debussy_file = "audios/sample1.wav"

debussy, sr = librosa.load(debussy_file)
sizeOfSignal = len(debussy)
FRAME_SIZE = 6144
HOP_LENGTH = int(FRAME_SIZE / 2)

def extractAmplitudeEnvelope(signal,  frame_size, hop_length):
    ae_signal = []
    for frame in range(0,len(signal),hop_length):
        current_frame_AE = (max(signal[frame:frame+frame_size]))
        ae_signal.append(current_frame_AE)
    return np.array(ae_signal)

def fancy_amplitude_envelope(signal, frame_size, hop_length):
    """Fancier Python code to calculate the amplitude envelope of a signal with a given frame size."""
    return np.array([max(signal[i:i+frame_size]) for i in range(0, len(signal), hop_length)])

def AEvsOrgPlot(original_signal, ae_signal, hop_length):
    frames = range(len(ae_signal))
    t = librosa.frames_to_time(frames, hop_length)
    plt.figure(figsize=(25,10))
    librosa.display.waveplot(original_signal)
    plt.plot(t,ae_signal,color='red')
    plt.savefig("AE.png")

# aeDebussy = extractAmplitudeEnvelope(signal=debussy,frame_size=FRAME_SIZE,hop_length=HOP_LENGTH)
aeDebussy = fancy_amplitude_envelope(signal=debussy,frame_size=FRAME_SIZE,hop_length=HOP_LENGTH)
AEvsOrgPlot(original_signal=debussy, ae_signal=aeDebussy,hop_length=HOP_LENGTH)
print("Done !")